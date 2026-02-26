# Align with search tools like openai/deepseek to be more Agentic.
import os
import re
import sys
import json
import math
import time
import fcntl
import random
import hashlib
import asyncio
import aiohttp
import textwrap
import pydantic
import tiktoken
import itertools
import functools
import threading
import traceback
import dataclasses
from datetime import datetime, timedelta
from transformers import AutoTokenizer, PreTrainedTokenizer
from dateutil.relativedelta import relativedelta
from urllib.parse import quote, unquote
from dacite import from_dict
from openai import AsyncOpenAI
from aiohttp import ClientError
from dataclasses import asdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union
from eas_prediction import PredictClient, ENDPOINT_TYPE_DIRECT, StringRequest
import html as html_escape


if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2])
    sys.path.append(base_path)

from utils.file_utils import FileUtils
from utils.logger import get_logger
from tools.types import WebContent, OrganicResults, SerpEntry, SerpProvider, WebProvider
from tools.cache import (
    WebCrawlCache,
    SearchCache,
)

from tools.search import (
    SearchToolBase,
    WebCrawlToolBase,
    execute_with_retries,
)
from utils.tools.simple_browser.simple_browser_tool import (
    SimpleBrowserState,
    FIND_PAGE_LINK_FORMAT, PARTIAL_INITIAL_LINK_PATTERN, PARTIAL_FINAL_LINK_PATTERN, LINK_PATTERN, CITATION_OUTPUT_PATTERN,
)
from utils.tools.simple_browser.page_contents import (
    PageContents,
    process_html,
    Extract
)
from utils.tools.simple_browser.backend import (
    maybe_truncate,
    VIEW_SOURCE_PREFIX,
)
logger = get_logger(__name__)

ENC_NAME_MAP = {
    "deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2"
}


class ToolUsageError(Exception):
    pass


@functools.cache
def _tokenizer_vocabulary_lengths(enc_name: str = "deepseek-v3.2") -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(ENC_NAME_MAP[enc_name])
    results = []
    n_vocab = len(tokenizer)
    for i in range(n_vocab):
        try:
            results.append(len(tokenizer.decode([i])))
        except:
            results.append(1)
    return results

@functools.cache
def _get_encoding(enc_name: str = "deepseek-v3.2") -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(ENC_NAME_MAP[enc_name])
    return tokenizer    

@dataclasses.dataclass(frozen=True)
class Tokens:
    tokens: list[int]
    tok2idx: list[int]  # Offsets = running sum of lengths.

@functools.cache
def max_chars_per_token(enc_name: str) -> int:
    """Typical value is 128, but let's be safe."""
    tok_lens = _tokenizer_vocabulary_lengths(enc_name)
    return max(tok_lens)

def get_tokens(text: str, enc_name: str) -> Tokens:
    #encoding = tiktoken.get_encoding(enc_name)
    encoding = _get_encoding(enc_name)
    tokens = encoding.encode(text, add_special_tokens=False)
    _vocabulary_lengths = _tokenizer_vocabulary_lengths(enc_name)
    tok2idx = [0] + list(itertools.accumulate(_vocabulary_lengths[i] for i in tokens))[
        :-1
    ]
    result = Tokens(tokens=tokens, tok2idx=tok2idx)
    return result

def join_lines(
    lines: list[str], add_line_numbers: bool = False, offset: int = 0
) -> str:
    if add_line_numbers:
        return "\n".join([f"L{i + offset}: {line}" for i, line in enumerate(lines)])
    else:
        return "\n".join(lines)

def get_end_loc(
    loc: int,
    num_lines: int,
    total_lines: int,
    lines: list[str],
    view_tokens: int,
    encoding_name: str,
) -> int:
    if num_lines <= 0:
        # COMPUTE NUMBER OF LINES TO SHOW
        txt = join_lines(lines[loc:], add_line_numbers=True, offset=loc)
        # if the text is very short, no need to truncate at all
        # at least one char per token
        if len(txt) > view_tokens:
            # limit the amount of text we tokenize here
            upper_bound = max_chars_per_token(encoding_name)
            tok2idx = get_tokens(
                txt[: (view_tokens + 1) * upper_bound], encoding_name
            ).tok2idx
            if len(tok2idx) > view_tokens:
                end_idx = tok2idx[view_tokens]
                num_lines = txt[:end_idx].count("\n") + 1  # round up
            else:
                num_lines = total_lines
        else:
            num_lines = total_lines

    return min(loc + num_lines, total_lines)

def get_page_metadata(
    curr_page: PageContents,
) -> dict[str, str | None | dict[str, str] | list[str]]:
    """Some attributes of the current page."""
    page_metadata: dict[str, str | None | dict[str, str] | list[str]] = {
        "url": curr_page.url,
        "title": curr_page.title,
    }
    return page_metadata

def wrap_lines(text: str, width: int = 80) -> list[str]:
    lines = text.split("\n")
    wrapped = itertools.chain.from_iterable(
        (
            textwrap.wrap(
                line, width=width, replace_whitespace=False, drop_whitespace=False
            )
            if line
            else [""]
        )  # preserve empty lines
        for line in lines
    )
    return list(wrapped)

def strip_links(text: str) -> str:
    text = re.sub(PARTIAL_INITIAL_LINK_PATTERN, "", text)
    text = re.sub(PARTIAL_FINAL_LINK_PATTERN, lambda mo: mo.group("content"), text)
    text = re.sub(LINK_PATTERN, lambda mo: mo.group("content"), text)
    return text

async def run_find_in_page(
    pattern: str,
    page: PageContents,
    max_results: int = 50,
    num_show_lines: int = 4,
) -> PageContents:
    lines = wrap_lines(text=page.text)
    txt = join_lines(lines, add_line_numbers=False)
    without_links = strip_links(txt)
    lines = without_links.split("\n")

    result_chunks, snippets = [], []
    line_idx, match_idx = 0, 0
    while line_idx < len(lines):
        line = lines[line_idx]
        if pattern not in line.lower():
            line_idx += 1
            continue
        snippet = "\n".join(lines[line_idx : line_idx + num_show_lines])
        link_title = FIND_PAGE_LINK_FORMAT.format(
            idx=f"{match_idx}", title=f"match at L{line_idx}"
        )
        result_chunks.append(f"{link_title}\n{snippet}")
        snippets.append(
            Extract(
                url=page.url, text=snippet, title=f"#{match_idx}", line_idx=line_idx
            )
        )
        if len(result_chunks) == max_results:
            break
        match_idx += 1
        line_idx += num_show_lines

    urls = [page.url for _ in result_chunks]

    if result_chunks:
        display_text = "\n\n".join(result_chunks)
    else:
        display_text = f"No `find` results for pattern: `{pattern}`"

    result_page = PageContents(
        url=f"{page.url}/find?pattern={quote(pattern)}",
        title=f"Find results for text: `{pattern}` in `{page.title}`",
        text=display_text,
        urls={str(i): url for i, url in enumerate(urls)},
        snippets={str(i): snip for i, snip in enumerate(snippets)},
    )
    return result_page


class SerperDevSearchToolV2:
    def __init__(self, config_path="config/config.yaml", **kwargs):
        _config = FileUtils.load_config(config_path)
        self.config_search = _config["tools"]["search"]
        self.config_crawl = _config["tools"]["crawl"]

        self.__serper_dev_web_search_url = "https://google.serper.dev/search"
        self.__serper_dev_news_search_url = "https://google.serper.dev/news"
        self.__serper_dev_web_crawl_url = "https://scrape.serper.dev"
        self.__jina_web_crawl_url = "https://r.jina.ai/"
        self.__serper_dev_token = os.getenv("SERPERDEV_TOKEN")
        self.__jina_token = os.getenv("JINA_TOKEN")

        self.web_cache = SearchCache(cache_dir=f"{self.config_search['cache_dir']}_web")
        self.news_cache = SearchCache(cache_dir=f"{self.config_search['cache_dir']}_news")
        self.crawl_cache = WebCrawlCache(cache_dir=f"{self.config_crawl['cache_dir']}")

    def filter_url(self, url: str) -> str:
        prefixes = ["https://r.jina.ai/", "view-source:"]
        for _prefix in prefixes:
            if url.startswith(_prefix):
                url = url[len(_prefix):]
                logger.info(f"Filtered URL: {url} by prefix {_prefix}")
                break
        return url

    async def search(self, query: str, topn: int = 10, source: str = "web") -> OrganicResults:
        # Do not add cache for now to reduce complexity of the initial version
        # if source == "web":
        #     cached_data = await asyncio.to_thread(self.web_cache.load, key=query)
        # elif source == "news":
        #     cached_data = await asyncio.to_thread(self.news_cache.load, key=query)
        # else:
        #     logger.warning(f"Invalid search source: {source}. Defaulting to web [Options: web, news].")
        #     cached_data = await asyncio.to_thread(self.web_cache.load, key=query)
        # if cached_data is not None: # cache hit
        #     # todo: cannot return here, need further processing
        #     return cached_data

        
        payload = {
            "q": query,
            "num": topn,
        }
        headers = {
            "X-API-KEY": self.__serper_dev_token,
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.config_search["timeout"])
        access_url = self.__serper_dev_web_search_url if source == "web" else self.__serper_dev_news_search_url
        async def _perform_serper_search() -> Dict[str, Any]:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        access_url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.status != 200:
                            body = await response.text()
                            raise RuntimeError(
                                f"Failed to search with SerperDev: {response.status} {body}"
                            )
                        return await response.json()
            except ClientError as exc:
                raise RuntimeError("SerperDev request failed") from exc

        try:
            results = await execute_with_retries(
                _perform_serper_search,
                context=f"serper.dev search for '{query}'",
                max_attempts=self.config_search["retry"]["max_attempts"],
                delay=self.config_search["retry"]["delay"],
            )
        except Exception as exc:
            logger.error(f"[SerperDevSearchToolV2.search] Serper.dev search failed for '{query}': {exc}")
            logger.error(traceback.format_exc())
            return OrganicResults(query=f"Failed to search for '{query}'", results=[])
        
        if source == "news":
            organic_results = results.get("news", [])
            title_and_urls = [
                (item.get("title", ""), item.get("link", ""), item.get("snippet", ""))
                for item in organic_results
            ]
        else: #source == "web":
            organic_results = results.get("organic", [])
            title_and_urls = [
                (item.get("title", ""), item.get("link", ""), item.get("snippet", ""))
                for item in organic_results
            ]
        html_page = f"""
<html><body>
<h1>Search Results</h1>
<ul>
{"".join([f"<li><a href='{url}'>{title}</a> {summary}</li>" for title, url, summary in title_and_urls])}
</ul>
</body></html>
"""
        return process_html(
            html=html_page,
            url="",
            title=query,
            display_urls=True,
            session=None,

        )
    
    async def backup_fetch(self, url: str):
        # Filter URL prefix
        url = self.filter_url(url)
        headers = {
            "X-API-KEY": self.__serper_dev_token,
            "Content-Type": "application/json",
        }
        payload = {"url": url}
        timeout = aiohttp.ClientTimeout(total=self.config_crawl["timeout"])

        async def _perform_serper_crawl() -> Dict[str, Any]:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.__serper_dev_web_crawl_url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.status != 200:
                            body = await response.text()
                            raise RuntimeError(
                                f"Failed to crawl with SerperDev: {response.status} {body}"
                            )
                        return await response.json()
            except ClientError as exc:
                raise RuntimeError("SerperDev crawl request failed") from exc

        try:
            results = await execute_with_retries(
                _perform_serper_crawl,
                context=f"serper.dev crawl for '{url}'",
                max_attempts=self.config_crawl["retry"]["max_attempts"],
                delay=self.config_crawl["retry"]["delay"],
            )
            assert results is not None
            assert results.get("text", None) is not None
        except (AssertionError, Exception) as exc:
            logger.error("[SerperDevSearchToolV2.backup_fetch] SerperDev crawl failed for %s: %s", url, exc)
            logger.error(traceback.format_exc())
            # Directly return error page
            return process_html(
                html="<html><body><p>Failed to access this page. The page content is unavailable.</p></body></html>",
                url=url,
                title="Error",
                display_urls=True,
                session=None,
            )

        # 1. Get plain text
        text = results.get("text", "")
        
        # 2. Extract title (from the first line)
        if text:
            title = text.split('\n')[0].strip()
        else:
            title = ""
        
        # 3. Convert plain text to HTML
        if text:
            # Escape HTML special characters
            escaped_text = html_escape.escape(text)
            
            # Split paragraphs by double newlines
            paragraphs = escaped_text.split('\n\n')
            
            # Process each paragraph: replace single newlines with <br> and wrap with <p> tags
            html_paragraphs = []
            for paragraph in paragraphs:
                # Replace single newlines with <br>
                paragraph_with_br = paragraph.replace('\n', '<br>')
                # Wrap with <p> tags (filter empty paragraphs)
                if paragraph_with_br.strip():
                    html_paragraphs.append(f'<p>{paragraph_with_br}</p>')
            
            # Assemble complete HTML
            html = f"<html><body>{''.join(html_paragraphs)}</body></html>"
        else:
            html = "<html><body><p>The page content is empty</p></body></html>"
        
        # 4. Process with process_html
        processed_result = process_html(
            html=html,
            url=url,
            title=title,
            display_urls=True,
            session=None,
        )
        
        return processed_result

    async def fetch(self, url: str):
        # Filter URL prefix
        url = self.filter_url(url)
        
        # Prepare Jina API request
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.__jina_token}",
            "Content-Type": "application/json",
            "X-Return-Format": "html",
            "X-Timeout": str(self.config_crawl["timeout"])
        }

        payload = {
            "url": url,
        }

        if self.config_crawl["jina"].get("token_budget", -1) != -1:
            headers["X-Token-Budget"] = str(self.config_crawl["jina"].get("token_budget", 200000))
        
        timeout = aiohttp.ClientTimeout(total=self.config_crawl["timeout"])

        async def _perform_jina_crawl() -> Dict[str, Any]:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.__jina_web_crawl_url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.status != 200:
                            body = await response.text()
                            raise RuntimeError(
                                f"Failed to crawl with Jina: {response.status} {body}"
                            )
                        return await response.json()
            except ClientError as exc:
                raise RuntimeError("Jina crawl request failed") from exc

        # Execute crawl request
        try:
            results = await execute_with_retries(
                _perform_jina_crawl,
                context=f"jina crawl for '{url}'",
                max_attempts=self.config_crawl["retry"]["max_attempts"],
                delay=self.config_crawl["retry"]["delay"],
            )
            assert results.get("data", None) is not None
            assert results.get("data", {}).get("html", None) is not None
            assert results.get("data", {}).get("html", "") != ""
        except (AssertionError, Exception) as exc:
            logger.error("[SerperDevSearchToolV2.fetch] Jina crawl failed for %s: %s", url, exc)
            logger.error(traceback.format_exc())
            # Directly call backup solution
            try:
                logger.info("[SerperDevSearchToolV2.fetch] Attempting backup fetch with SerperDev for %s", url)
                return await self.backup_fetch(url)
            except Exception as backup_exc:
                logger.error("[SerperDevSearchToolV2.fetch] Backup fetch also failed for %s: %s", url, backup_exc)
                logger.error(traceback.format_exc())
                # Return an error page
                return process_html(
                    html="<html><body><p>Failed to access this page. The page content is unavailable.</p></body></html>",
                    url=url,
                    title="Error",
                    display_urls=True,
                    session=None,
                )

        # Process returned data
        data = results.get("data", {})
        html = data.get("html", "")
        processed_result = process_html(
            html=html,
            url=url,
            title=data.get("title", ""),
            display_urls=True,
            session=None,
        )
        
        return processed_result


class DeepSeekBrowseTool:
    def __init__(self, backend: SerperDevSearchToolV2, encoding_name: str = "deepseek-v3.2", max_search_results: int = 10, tool_state: dict[str, Any] = None, view_tokens: int = 1024):
        self.backend = backend
        if tool_state is None:
            self.tool_state = SimpleBrowserState()
        else:
            self.tool_state = SimpleBrowserState.model_validate(tool_state)
        
        self.encoding_name = encoding_name
        self.max_search_results = max_search_results
        self.view_tokens = view_tokens
    
    def get_tool_state(self) -> dict[str, Any]:
        return {"tool_state": self.tool_state.model_dump()}
    
    def _render_browsing_display(
        self,
        tether_id: int,
        result: str,
        summary: str | None = None,
    ):
        to_return = ""
        # Always show summaries.
        if summary:
            to_return += summary
        to_return += result
        to_return = f"[{tether_id}] {to_return}"
        return to_return

    async def show_page_safely(self, loc: int = 0, num_lines: int = -1):
        try:
            return await self.show_page(loc=loc, num_lines=num_lines)
        except ToolUsageError as e:
            self.tool_state.pop_page_stack()
            raise e

    async def show_page(self, loc: int = 0, num_lines: int = -1):
        page = self.tool_state.get_page()
        cursor = self.tool_state.current_cursor
        lines = wrap_lines(text=page.text)
        total_lines = len(lines)

        if loc >= total_lines:
            err_msg = (
                f"Invalid location parameter: `{loc}`. "
                f"Cannot exceed page maximum of {total_lines - 1}."
            )
            raise ToolUsageError(err_msg)

        end_loc = get_end_loc(
            loc, num_lines, total_lines, lines, self.view_tokens, self.encoding_name
        )

        lines_to_show = lines[loc:end_loc]
        body = join_lines(lines_to_show, add_line_numbers=True, offset=loc)

        scrollbar = f"viewing lines [{loc} - {end_loc - 1}] of {total_lines - 1}"
        return self._make_response(page, cursor, body, scrollbar)

    def _make_response(
        self,
        page: PageContents,
        cursor: int,
        body: str,
        scrollbar: str,
    ):
        domain = maybe_truncate(unquote(page.url))
        header = f"{page.title}"
        if domain:
            header += f" ({domain})"
        header += f"\n**{scrollbar}**\n\n"

        return self._render_browsing_display(cursor, body, header)

        # content = TextContent(text=self._render_browsing_display(cursor, body, header))
        # return self.make_response(
        #     content=content, metadata=get_page_metadata(self.tool_state.get_page())
        # )

    async def search(self, query: str, topn: int = 10, source: str = "web"):
        try:
            search_page = await self.backend.search(
                query=query,
                topn=topn,
                source=source
            )
        except Exception as e:
            logger.error("[DeepSeekBrowseTool.search] Search failed for %s: %s", query, e)
            logger.error(traceback.format_exc())
            raise RuntimeError(f"[DeepSeekBrowseTool.search] Error during search for `{query}`: {e}")
        
        
        self.tool_state.add_page(search_page)

        return await self.show_page_safely(loc=0)
    
    async def _open_url(self, url: str, direct_url_open: bool) -> PageContents:
        """Use the cache, if available."""
        # direct_url_open should be regarded as a refresh
        if not direct_url_open and (page := self.tool_state.get_page_by_url(url)):
            assert page.url == url
            return page

        try:
            #async with aiohttp.ClientSession() as session:
            #    page = await self.backend.fetch(url, session=session)
            page = await self.backend.fetch(url)
            return page
        except Exception as e:
            msg = maybe_truncate(str(e))
            logger.warning("Error fetching URL in lean browser tool", exc_info=e)
            raise RuntimeError(
                f"Error fetching URL `{maybe_truncate(url)}`: {msg}"
            ) from e

    async def open(self, id: int | str = -1, cursor: int = -1, loc: int = -1, num_lines: int = -1, view_source: bool = False, source: str | None = None):
        stay_on_current_page = False
        direct_url_open = False
        if isinstance(id, str):
            snippet = None
            url = id
            direct_url_open = True
        else:  # Operate on a previously opened page
            curr_page = self.tool_state.get_page(cursor)

            if id >= 0:  # click a link
                try:
                    url = curr_page.urls[str(id)]
                except KeyError as e:
                    raise ToolUsageError(f"Invalid link id `{id}`.") from e
                snippet = (curr_page.snippets or {}).get(str(id))
                if snippet and curr_page.url == "":
                    # current page is a search result page
                    assert isinstance(snippet, Extract)
            else:  # navigate to new position on the current page
                if not view_source:
                    stay_on_current_page = True
                url = curr_page.url
                snippet = None

        new_page: PageContents
        if view_source:
            url = f"{VIEW_SOURCE_PREFIX}{url}"
            snippet = None
        if stay_on_current_page:
            assert curr_page is not None
            new_page = curr_page
        else:
            new_page = await self._open_url(url, direct_url_open)

        self.tool_state.add_page(new_page)

        if loc < 0:  # unset
            if snippet is not None and snippet.line_idx is not None:
                loc = snippet.line_idx
                if loc > 4:
                    loc -= 4
            else:
                loc = 0
        
        return await self.show_page_safely(loc=loc, num_lines=num_lines)

    async def find(self, pattern: str, cursor: int = -1):
        page = self.tool_state.get_page(cursor)
        if page.snippets is not None:
            raise ToolUsageError("Cannot run `find` on search results page or find results page")
        
        pc = await run_find_in_page(
            pattern=str(pattern).lower(),
            page=page
        )
        self.tool_state.add_page(pc)
        return await self.show_page_safely(loc=0)

if __name__ == "__main__":
    async def test_deepseek_browse_tool():
        """测试 DeepSeekBrowseTool 的基本功能"""
        print("=" * 80)
        print("开始测试 DeepSeekBrowseTool")
        print("=" * 80)
        
        # 1. 初始化 backend
        print("\n[1] 初始化 SerperDevSearchToolV2...")
        config_path = "config/config.yaml"  # 根据你的实际路径修改
        backend = SerperDevSearchToolV2(config_path=config_path)
        print("✓ Backend 初始化成功")
        
        # 2. 初始化 DeepSeekBrowseTool
        print("\n[2] 初始化 DeepSeekBrowseTool...")
        browse_tool = DeepSeekBrowseTool(
            backend=backend,
            encoding_name="deepseek-v3.2",  # 使用 OpenAI 的标准编码
            max_search_results=10,
            tool_state=None,
            view_tokens=1024
        )
        print("✓ DeepSeekBrowseTool 初始化成功")
        
        # 3. 测试搜索功能
        print("\n[3] 测试搜索功能...")
        try:
            query = "Python asyncio tutorial"
            print(f"搜索查询: {query}")
            search_result = await browse_tool.search(query=query, topn=5, source="web")
            print(f"✓ 搜索成功")
            print(f"搜索结果预览 (前500字符):\n{search_result[:500]}")
            print(f"...\n")
        except Exception as e:
            print(f"✗ 搜索失败: {e}")
            traceback.print_exc()
        
        # 4. 测试打开链接功能（使用搜索结果中的第一个链接）
        print("\n[4] 测试打开链接功能...")
        try:
            # 打开搜索结果中的第一个链接
            open_result = await browse_tool.open(id=0, loc=0, num_lines=20)
            print(f"✓ 打开链接成功")
            print(f"页面内容预览 (前500字符):\n{open_result[:500]}")
            print(f"...\n")
        except Exception as e:
            print(f"✗ 打开链接失败: {e}")
            traceback.print_exc()
        
        # 5. 测试直接打开 URL
        print("\n[5] 测试直接打开 URL...")
        try:
            test_url = "https://docs.python.org/3/library/asyncio.html"
            print(f"打开 URL: {test_url}")
            url_result = await browse_tool.open(url=test_url, id=test_url, loc=0, num_lines=30)
            print(f"✓ 直接打开 URL 成功")
            print(f"页面内容预览 (前500字符):\n{url_result[:500]}")
            print(f"...\n")
        except Exception as e:
            print(f"✗ 直接打开 URL 失败: {e}")
            traceback.print_exc()
        
        # 6. 测试获取 tool state
        print("\n[6] 测试获取 tool state...")
        try:
            tool_state = browse_tool.get_tool_state()
            print(f"✓ 获取 tool state 成功")
            print(f"Tool state keys: {tool_state.keys()}")
            if "tool_state" in tool_state:
                inner_state = tool_state["tool_state"]
                print(f"Inner state keys: {inner_state.keys() if isinstance(inner_state, dict) else type(inner_state)}")
        except Exception as e:
            print(f"✗ 获取 tool state 失败: {e}")
            traceback.print_exc()
        
        # 7. 测试 Backend 的 fetch 方法
        print("\n[7] 测试 Backend 的 fetch 方法...")
        try:
            test_url = "https://www.example.com"
            print(f"Fetch URL: {test_url}")
            fetch_result = await backend.fetch(url=test_url)
            print(f"✓ Fetch 成功")
            print(f"Fetch 结果类型: {type(fetch_result)}")
            if hasattr(fetch_result, 'text'):
                print(f"页面文本预览 (前300字符):\n{fetch_result.text[:300]}")
            elif isinstance(fetch_result, str):
                print(f"页面文本预览 (前300字符):\n{fetch_result[:300]}")
            print(f"...\n")
        except Exception as e:
            print(f"✗ Fetch 失败: {e}")
            traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("测试完成")
        print("=" * 80)
    
    # 运行测试
    print("启动异步测试...")
    asyncio.run(test_deepseek_browse_tool())