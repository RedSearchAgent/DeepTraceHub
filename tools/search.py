from nturl2path import url2pathname
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
import tiktoken
import threading
import traceback
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


from dacite import from_dict
from openai import AsyncOpenAI
from aiohttp import ClientError
from dataclasses import asdict
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union
from eas_prediction import PredictClient, ENDPOINT_TYPE_DIRECT, StringRequest


if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2])
    sys.path.append(base_path)

from utils.file_utils import FileUtils
from utils.logger import get_logger
from tools.types import WebContent, WebContentV2, OrganicResults, SerpEntry, SerpProvider, WebProvider
from tools.cache import (
    WebCrawlCacheV2 as WebCrawlCache,
    SearchCache,
)

logger = get_logger(__name__)

T = TypeVar("T")

def contains_chinese_basic(text: str) -> bool:
    return any('\u4E00' <= char <= '\u9FFF' for char in text)

def convert_format_date(format_date: str) -> str:
    """
    Convert format date to natural language.

    Args:
        format_date: Format date string, e.g. "2025-09-19T03:15:00"

    Returns:
        str: Natural language date string, e.g. "Sep 19, 2025"
    """
    try:
        dt = datetime.fromisoformat(format_date.replace('Z', '+00:00'))
        return dt.strftime("%b %d, %Y")
    except Exception as e:
        return format_date

def convert_date(date_desc: str) -> str:
    """
    Convert date descriptions to absolute date format.
    
    Args:
        date_desc: Date description string, either relative time (e.g., "2 days ago") 
                  or absolute time (e.g., "Sep 1, 2025")
    
    Returns:
        str: Formatted absolute date string
    """
    if not date_desc or not isinstance(date_desc, str):
        return date_desc
    date_desc = date_desc.strip()
    
    # Check if it's a relative time pattern
    relative_pattern = r'(\d+)\s+(hour|hours|day|days|week|weeks|month|months|year|years)\s+ago'
    match = re.match(relative_pattern, date_desc, re.IGNORECASE)
    if match:
        amount = int(match.group(1))
        unit = match.group(2).lower()
        
        current_date = datetime.now()
        if unit in ['hour', 'hours']:
            # For hours, return current date (Month Day, Year)
            target_date = current_date - timedelta(hours=amount)
            return target_date.strftime("%b %d, %Y")

        elif unit in ['day', 'days']:
            # For days, return (Month Day, Year)
            target_date = current_date - timedelta(days=amount)
            return target_date.strftime("%b %d, %Y")
            
        elif unit in ['week', 'weeks']:
            # For weeks, return (Month Day, Year)
            target_date = current_date - timedelta(weeks=amount)
            return target_date.strftime("%b %d, %Y")
            
        elif unit in ['month', 'months']:
            # For months, return (Month, Year)
            target_date = current_date - relativedelta(months=amount)
            return target_date.strftime("%b, %Y")
            
        elif unit in ['year', 'years']:
            # For years, return (Year)
            target_date = current_date - relativedelta(years=amount)
            return target_date.strftime("%Y")
    # If it's not a relative time pattern, return the original string (absolute time)
    return date_desc

async def execute_with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 2,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    context: str = "operation",
) -> T:
    """Run `operation` with retry and exponential backoff using async sleep."""
    attempt = 0
    current_delay = delay
    last_exc: Optional[Exception] = None

    while attempt < max_attempts:
        try:
            return await operation()
        except Exception as exc:  # noqa: BLE001 - retry helper needs to catch broadly by design
            last_exc = exc
            attempt += 1
            logger.warning(
                "Attempt %s/%s failed for %s: %s",
                attempt,
                max_attempts,
                context,
                exc,
            )
            if attempt >= max_attempts:
                break
            jitter = random.uniform(0.8, 1.2)
            await asyncio.sleep(max(current_delay, 0) * jitter)
            current_delay *= backoff_factor

    if last_exc is not None:
        raise last_exc

    raise RuntimeError(f"Retry helper encountered an unexpected state for {context}")

class SerperDevKeyManager:
    """API key manager for SerperDev, used to manage API key usage.
    Since the current keys are free, each has only 2500 credits, so they need to be used in rotation. 
    Once a paid Serper key is purchased, this manager will no longer be needed."""
    def __init__(self, usage_file_path: str):

        self.usage_file_path = usage_file_path
        # File lock to protect concurrent writes to serper_key_usage.json
        self._file_lock = threading.Lock()
        self._load_usage_data()
    
    def _load_usage_data(self):
        """Load usage data from serper_key_usage.json, keys are retrieved from this file"""

        if not self.usage_file_path or not os.path.exists(self.usage_file_path):
            logger.warning(f"Usage file {self.usage_file_path} not found. We use an empty usage data.")
            self.usage_data, self.keys = {}, []
            return
        try:
            self.usage_data = FileUtils.load_json(self.usage_file_path)
            # Get all keys from usage_data
            self.keys = list(self.usage_data.keys())
            logger.info(f"Loaded {len(self.keys)} API keys from usage file")
        except Exception as e:
            logger.error(f"Error loading usage data: {e}")
            self.usage_data = {}
            self.keys = []
    
    def _save_usage_data(self, key: str, credits: Union[int, str, None] = None):
        """Save usage data (with file lock, supports multi-process)"""
        # Use file lock to ensure atomic writes in multi-process environments
        try:
            credits = int(credits)
        except Exception as e:
            credits = 2
            logger.warning(f"Error converting credits to int: {e}. We use 2 as default credits.")
        lock_file_path = self.usage_file_path + '.lock'
        delays = [0.1, 0.2, 0.3, 0.5, 0.7, 1.1]
        max_retries = len(delays)
        
        for attempt in range(max_retries):
            try:
                # Use a separate lock file
                with open(lock_file_path, 'w') as lock_file:
                    # Non-blocking lock, retry if unable to acquire
                    try:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except IOError:
                        # Lock held by another process, wait and retry
                        if attempt < max_retries - 1:
                            time.sleep(delays[attempt] * random.uniform(0.8, 1.2))  # Exponential backoff
                            continue
                        else:
                            # Retry limit reached, still cannot acquire lock, skip write
                            logger.warning("Failed to acquire lock after all retries, skipping write to usage file")
                            return
                    
                    try:
                        # Reload data to prevent missing changes from other processes
                        self._load_usage_data()
                        
                        # Atomic write: write to temporary file first, then move
                        temp_file_path = self.usage_file_path + '.tmp'
                        with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
                            self.usage_data[key] = max(0, self.usage_data[key] - credits)
                            json.dump(self.usage_data, temp_file, indent=4, ensure_ascii=False)
                            temp_file.flush()
                            os.fsync(temp_file.fileno())  # Force write to disk
                        
                        # Atomic move (atomic operation on most file systems)
                        os.replace(temp_file_path, self.usage_file_path)
                        
                    finally:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                
                # Successful write, break retry loop
                break
                
            except Exception as e:
                logger.error(f"Error saving usage data (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error("Failed to save usage data after all retries")
                else:
                    time.sleep(delays[attempt] * random.uniform(0.8, 1.2))
            finally:
                # Clean up lock file
                try:
                    if os.path.exists(lock_file_path):
                        os.unlink(lock_file_path)
                except:
                    pass
    
    def select_key(self) -> Optional[str]:
        """
        Select API key based on remaining credits with probability
        """
        # Filter keys with remaining credits
        available_keys = []
        weights = []
        
        for key in self.keys:
            if key in self.usage_data and self.usage_data[key] > 0:
                available_keys.append(key)
                weights.append(self.usage_data[key])
        
        if not available_keys:
            logger.warning("No available keys with remaining credits")
            return None
        
        # Random choice based on weights
        try:
            selected_key = random.choices(available_keys, weights=weights, k=1)[0]
            logger.debug(f"Selected key: {selected_key[:8]}... with {self.usage_data[selected_key]} credits")
            return selected_key
        except Exception as e:
            logger.error(f"Error selecting key: {e}")
            return available_keys[0] if available_keys else None
    
    def update_api_key_credits(self, api_key: str, credits: Union[int, str, None] = None):
        """
        Update the number of credits for an API key
        
        Args:
            api_key: API key
            credits: Number of credits consumed this time
        """
        if api_key in self.usage_data:
            #self.usage_data[api_key] = max(0, self.usage_data[api_key] - credits)
            self._save_usage_data(api_key, credits)
        else:
            logger.warning(f"API key {api_key[:8]}... not found")

class SearchToolBase:
    def __init__(self, config_path="config/config.yaml", **kwargs):
        self.config = FileUtils.load_config(
            config_path
        )["tools"]["search"]

    async def search(self, query: str) -> OrganicResults:
        raise NotImplementedError("Subclass must implement search method: SearchToolBase.search")

class WebCrawlToolBase:
    def __init__(self, config_path="config/config.yaml", **kwargs):
        self.config = FileUtils.load_config(
            config_path
        )["tools"]["crawl"]
        self.cache = WebCrawlCache(cache_dir=self.config["cache_dir"])

    async def crawl(self, url: str) -> Union[WebContent, WebContentV2]:
        raise NotImplementedError("Subclass must implement crawl method: WebCrawlToolBase.crawl")

    def filter_url(self, url: str) -> str:
        prefixes = ["https://r.jina.ai/", "view-source:"]
        for _prefix in prefixes:
            if url.startswith(_prefix):
                url = url[len(_prefix):]
                logger.info(f"Filtered URL: {url} by prefix {_prefix}")
                break
        return url

class SerperDevSearchTool(SearchToolBase):
    name = "serper.dev"
    category = 'search'
    def __init__(self, config_path="config/config.yaml", **kwargs):
        super().__init__(config_path, **kwargs)
        self.__key_manager = kwargs.get("key_manager", None)
        self.__serper_dev_url = "https://google.serper.dev/search"
        self.__serper_dev_token = os.getenv("SERPERDEV_TOKEN")
        self.cache = SearchCache(cache_dir=f"{self.config['cache_dir']}/{self.name}")

    async def search(self, query: str) -> OrganicResults:
        cached_data = await asyncio.to_thread(self.cache.load, key=query)
        if cached_data is not None: # cache hit
            return cached_data
        
        #serper_api_key = self.__key_manager.select_key()
        serper_api_key = self.__serper_dev_token
        default_credits_used = math.ceil(self.config["num"] / 10)
        payload = {
            "q": query,
            "num": self.config["num"],
        }
        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.config["timeout"])

        async def _perform_serper_search() -> Dict[str, Any]:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.__serper_dev_url,
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
                max_attempts=self.config["retry"]["max_attempts"],
                delay=self.config["retry"]["delay"],
            )
        except Exception as exc:
            logger.error(f"[SerperDevSearchTool.search] Serper.dev search failed for '{query}': {exc}")
            logger.error(traceback.format_exc())
            return OrganicResults(query=f"Failed to search for '{query}'", results=[])
            raise RuntimeError(f"[SerperDevSearchTool.search] Serper.dev search failed after retries: {exc}") from exc

        serps: List[SerpEntry] = []
        for idx, result in enumerate(results["organic"]):
            serps.append(
                SerpEntry(
                    url=result.get("link", ""),
                    title=result.get("title", ""),
                    snippet=result.get("snippet", ""),
                    content=None,
                    rank=idx,
                    source=SerpProvider.SERPER,
                    raw_organic=result
                )
            )
        final_organic_results = OrganicResults(query=query, results=serps)
        await asyncio.to_thread(self.cache.save, key=query, value=final_organic_results)
        return final_organic_results

class EmptyWebSearchTool(SearchToolBase):
    name = "empty"
    category = "search"
    def __init__(self, config_path="config/config.yaml", **kwargs):
        super().__init__(config_path, **kwargs)
    
    async def search(self, query: str) -> OrganicResults:
        return OrganicResults(query=query, results=[])

class LocalWebSearchTool(SearchToolBase):
    name = "local"
    category = "search"
    # Local Search does not need cache.
    def __init__(self, config_path="config/config.yaml", **kwargs):
        super().__init__(config_path, **kwargs)
        self.__local_web_search_url = os.getenv("LOCAL_SEARCH_URL")
    
    async def search(self, query: str) -> OrganicResults:
        payload = {"query": query, "k": self.config["num"]}
        headers = {"Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=self.config["timeout"])
        async def _perform_local_search() -> Dict[str, Any]:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.__local_web_search_url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.status != 200:
                            body = await response.text()
                            raise RuntimeError(
                                f"Failed to search with Local Search: {response.status} {body}"
                            )
                        return await response.json()
            except ClientError as exc:
                raise RuntimeError("SerperDev request failed") from exc    
        
        try:
            results = await execute_with_retries(
                _perform_local_search,
                context=f"local search for '{query}'",
                max_attempts=self.config["retry"]["max_attempts"],
                delay=self.config["retry"]["delay"],
            )
        except Exception as exc:
            logger.error(f"[LocalWebSearchTool.search] Local search failed for '{query}': {exc}")
            logger.error(traceback.format_exc())
            return OrganicResults(query=f"Failed to search for '{query}'", results=[])
        
        web_content_list: List[WebContent] = []
        serps: List[SerpEntry] = []
        for idx, result in enumerate(results):
            title, text, url, abstract = result.get("title", ""), result.get("text", ""), result.get("synthesis_url", ""), result.get("abstract", "")
            # Current issue: the returned text is too long, leading to exceeding limits within 100 rounds.
            # Need to process abstract.
            if contains_chinese_basic(abstract):
                snippet = abstract[:25]
            else:
                words = abstract.split()
                snippet = " ".join(words[:40])
            serps.append(
                SerpEntry(
                    url=url,
                    title=title,
                    snippet=snippet,
                    content=None,
                    rank=idx,
                    source=SerpProvider.LOCAL,
                    raw_organic=result
                )
            )
            web_content_list.append(WebContent(
                text_content=text,
                source=WebProvider.LOCAL
            ))
            
        final_organic_results = OrganicResults(query=query, results=serps)
        return final_organic_results

class LocalWebCrawlTool(WebCrawlToolBase):
    name = "local"
    category = "crawl"
    def __init__(self, config_path="config/config.yaml", **kwargs):
        super().__init__(config_path, **kwargs)
        self.__local_web_crawl_url = os.getenv("LOCAL_CRAWL_URL")
        self.cache = WebCrawlCache(cache_dir=f"{self.config['cache_dir']}")

    async def crawl(self, url: str) -> Union[WebContent, WebContentV2]:
        url = self.filter_url(url)
        cached_data = await asyncio.to_thread(self.cache.load, key=url)
        if cached_data:
            return cached_data
        
        headers = {"Content-Type": "application/json"}
        payload = {"url": url}
        timeout = aiohttp.ClientTimeout(total=self.config["timeout"])

        async def _perform_local_crawl() -> str:
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.__local_web_crawl_url,
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.status != 200:
                            body = await response.text()
                            raise RuntimeError(
                                f"Failed to crawl with Local: {response.status} {body}"
                            )
                        return await response.json()
            except ClientError as exc:
                raise RuntimeError("Local crawl request failed") from exc

        try:
            results = await execute_with_retries(
                _perform_local_crawl,
                context=f"local crawl for '{url}'",
                max_attempts=self.config["retry"]["max_attempts"],
                delay=self.config["retry"]["delay"],
            )
            assert results.get("success")
        except Exception as exc:
            logger.error("[LocalWebCrawlTool.crawl] Local crawl failed for %s: %s", url, exc)
            logger.error(traceback.format_exc())
            return WebContentV2(
                url=url,
                title=None,
                snippet=None,
                content=WebContent(
                    text_content=None,
                    source=WebProvider.LOCAL
                ),
                rank=0,
                source=WebProvider.LOCAL,
                raw_organic=None
            )

        results = results.get("result", {})
        if "wikipedia_id" in results:
            web_content = WebContent(
                text_content=results.get("text"),
                source=WebProvider.LOCAL,
                #from_cache=False
            )
            final_web_content = WebContentV2(
                url=results.get("synthesis_url"),
                title=results.get("wikipedia_title"),
                snippet=results.get("abstract"),
                content=web_content,
                rank=0,
                source=SerpProvider.LOCAL,
                raw_organic=None,
                #from_cache=False
            )
        # case2: real data
        else:
            web_content = WebContent(
                text_content=results.get("content"),
                source=WebProvider.LOCAL,
                #from_cache=False
            )
            final_web_content = WebContentV2(
                url=results.get("url"),
                title=results.get("title"),
                snippet=results.get("snippet"),
                content=web_content,
                rank=0,
                source=SerpProvider.LOCAL,
                raw_organic=None,
                #from_cache=False
            )
        
        
        return final_web_content
        
class SerperDevWebCrawlTool(WebCrawlToolBase):
    name = "serper.dev"
    category = 'crawl'
    def __init__(self, config_path="config/config.yaml", **kwargs):
        super().__init__(config_path, **kwargs)
        self.__serper_dev_web_crawl_url = "https://scrape.serper.dev"
        self.__serper_dev_web_crawl_token = os.getenv("SERPERDEV_TOKEN")
        self.__key_manager = kwargs.get("key_manager", None)

    async def crawl(self, url: str) -> Union[WebContent, WebContentV2]:
        cached_data = await asyncio.to_thread(self.cache.load, key=url)
        if cached_data is not None: # cache hit
            return cached_data
        
        url = self.filter_url(url)
        
        #serper_api_key = self.__key_manager.select_key()
        serper_api_key = self.__serper_dev_web_crawl_token
        payload = {
            "url": url,
        }
        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.config["timeout"])

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
                max_attempts=self.config["retry"]["max_attempts"],
                delay=self.config["retry"]["delay"],
            )
        except Exception as exc:
            logger.error("[SerperDevWebCrawlTool.crawl] SerperDev crawl failed for %s: %s", url, exc)
            return WebContent(
                text_content=None,
                source=WebProvider.SERPER,
            )

        metadata = results.get("metadata", {})
        web_content = WebContent(
            text_content=results.get("text", None),
            source=WebProvider.SERPER,
        )
        title = metadata.get("title", "")
        description = metadata.get("description", "")
        if description == title or not description:
            snippet = None
        else:
            snippet = description
        final_web_content = WebContentV2(
            url=url,
            title=title,
            snippet=snippet,
            content=web_content,
            rank=0,
            source=SerpProvider.SERPER,
            raw_organic=None
        )
        await asyncio.to_thread(self.cache.save, key=url, value=final_web_content)
        return final_web_content

class TavilySearchTool(SearchToolBase):
    name = "tavily"
    category = "search"
    def __init__(self, config_path="config/config.yaml", **kwargs):
        super().__init__(config_path, **kwargs)
        self.cache = SearchCache(cache_dir=f"{self.config['cache_dir']}/{self.name}")

    async def search(self, query: str) -> OrganicResults:
        from tavily import AsyncTavilyClient

        cached_data = await asyncio.to_thread(self.cache.load, key=query)
        if cached_data is not None:
            return cached_data

        tavily_client = AsyncTavilyClient()
        max_results = self.config.get("num", 10)

        async def _perform_tavily_search():
            return await tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
            )

        try:
            results = await execute_with_retries(
                _perform_tavily_search,
                context=f"tavily search for '{query}'",
                max_attempts=self.config["retry"]["max_attempts"],
                delay=self.config["retry"]["delay"],
            )
        except Exception as exc:
            logger.error(f"[TavilySearchTool.search] Tavily search failed for '{query}': {exc}")
            logger.error(traceback.format_exc())
            return OrganicResults(query=f"Failed to search for '{query}'", results=[])

        serps: List[SerpEntry] = []
        for idx, result in enumerate(results.get("results", [])):
            serps.append(
                SerpEntry(
                    url=result.get("url", ""),
                    title=result.get("title", ""),
                    snippet=result.get("content", ""),
                    content=None,
                    rank=idx,
                    source=SerpProvider.TAVILY,
                    raw_organic=result,
                )
            )
        final_organic_results = OrganicResults(query=query, results=serps)
        await asyncio.to_thread(self.cache.save, key=query, value=final_organic_results)
        return final_organic_results


class TavilyWebCrawlTool(WebCrawlToolBase):
    name = "tavily"
    category = "crawl"
    def __init__(self, config_path="config/config.yaml", **kwargs):
        super().__init__(config_path, **kwargs)

    async def crawl(self, url: str) -> Union[WebContent, WebContentV2]:
        from tavily import AsyncTavilyClient

        cached_data = await asyncio.to_thread(self.cache.load, key=url)
        if cached_data is not None:
            return cached_data

        url = self.filter_url(url)
        tavily_client = AsyncTavilyClient()

        async def _perform_tavily_extract():
            return await tavily_client.extract(urls=[url])

        try:
            results = await execute_with_retries(
                _perform_tavily_extract,
                context=f"tavily extract for '{url}'",
                max_attempts=self.config["retry"]["max_attempts"],
                delay=self.config["retry"]["delay"],
            )
        except Exception as exc:
            logger.error("[TavilyWebCrawlTool.crawl] Tavily extract failed for %s: %s", url, exc)
            logger.error(traceback.format_exc())
            return WebContentV2(
                url=url,
                title=None,
                snippet=None,
                content=WebContent(
                    text_content=None,
                    source=WebProvider.TAVILY
                ),
                rank=0,
                source=WebProvider.TAVILY,
                raw_organic=None,
            )

        extracted_results = results.get("results", [])
        if extracted_results:
            first = extracted_results[0]
            raw_content = first.get("raw_content", "")
            web_content = WebContent(
                text_content=raw_content,
                source=WebProvider.TAVILY,
            )
            final_web_content = WebContentV2(
                url=first.get("url", url),
                title=None,
                snippet=None,
                content=web_content,
                rank=0,
                source=WebProvider.TAVILY,
                raw_organic=first,
            )
        else:
            final_web_content = WebContentV2(
                url=url,
                title=None,
                snippet=None,
                content=WebContent(
                    text_content=None,
                    source=WebProvider.TAVILY,
                ),
                rank=0,
                source=WebProvider.TAVILY,
                raw_organic=None,
            )

        await asyncio.to_thread(self.cache.save, key=url, value=final_web_content)
        return final_web_content


class JinaWebCrawlTool(WebCrawlToolBase):
    # jina crawl docs: https://r.jina.ai/docs#tag/crawl/paths/~1%7Burl%7D/get
    name = "jina"
    category = 'crawl'
    def __init__(self, config_path="config/config.yaml", **kwargs):
        super().__init__(config_path, **kwargs)
        self.__jina_web_crawl_url = "https://r.jina.ai/"
        self.__jina_web_crawl_token = os.getenv("JINA_TOKEN")
        
    async def crawl(self, url: str) -> Union[WebContent, WebContentV2]:
        cached_data = await asyncio.to_thread(self.cache.load, key=url)
        if cached_data is not None: # cache hit
            return cached_data
        
        headers = {
            "Authorization": f"Bearer {self.__jina_web_crawl_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Proxy": "auto"
        }

        # Tongyi's training data was contaminated, so we implement a filter.
        url = self.filter_url(url)

        payload = {
            "url": url,
        }

        if self.config["jina"].get("token_budget", -1) != -1:
            payload["tokenBudget"] = self.config["jina"].get("token_budget", 200000)
        timeout = aiohttp.ClientTimeout(total=self.config["timeout"])

        async def _perform_jina_crawl() -> str:
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

        try:
            results = await execute_with_retries(
                _perform_jina_crawl,
                context=f"jina crawl for '{url}'",
                max_attempts=self.config["retry"]["max_attempts"],
                delay=self.config["retry"]["delay"],
            )
        except Exception as exc:
            logger.error("[JinaWebCrawlTool.crawl] Jina crawl failed for %s: %s", url, exc)
            logger.error(traceback.format_exc())
            return WebContentV2(
                url=url,
                title=None,
                snippet=None,
                content=WebContent(
                    text_content=None,
                    source=WebProvider.JINA
                ),
                rank=0,
                source=WebProvider.JINA,
                raw_organic=None
            )

        data = results.get("data", {})
        web_content = WebContent(
            text_content=data.get("content", ""),
            source=WebProvider.JINA,
        )
        title = data.get("title", "")
        description = data.get("description", "")
        if description == title or not description:
            snippet = None
        else:
            snippet = description
        final_web_content = WebContentV2(
            url=url,
            title=title,
            snippet=snippet,
            content=web_content,
            rank=0,
            source=WebProvider.JINA,
            raw_organic=None
        )
        await asyncio.to_thread(self.cache.save, key=url, value=final_web_content)
        return final_web_content


