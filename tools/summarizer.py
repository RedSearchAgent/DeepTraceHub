import os

import sys
import json
import asyncio
import tiktoken
from openai import AsyncOpenAI

if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2])
    sys.path.append(base_path)

from utils.file_utils import FileUtils
from utils.logger import get_logger
from tools.search import execute_with_retries
from prompt.system_prompt import EXTRACTOR_PROMPT, EXTRACTOR_PROMPT_BRIEF

logger = get_logger(__name__)

def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    """
    Use cl100k_base encoder to truncate any text to the specified number of tokens,
    avoiding subsequent model input being too long.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

class SummarizerBase:
    def __init__(self, config_path="config/config.yaml"):
        self.config = FileUtils.load_config(config_path)

    async def summarize(self, url: str, raw_content: str, goal: str) -> str:
        raise NotImplementedError("Subclass must implement summarize method: Summarizer.summarize")


class LLMSummarizer(SummarizerBase):
    """
    Summarize web content with given goal.
    """
    name = "llm_summarizer"
    category = "summarizer"
    def __init__(self, config_path="config/config.yaml"):
        super().__init__(config_path)
        self.client = AsyncOpenAI(
            base_url = self.config["tools"]["summarize"]["endpoint"],
            api_key = "abc"
        )
        self.model_name = self.config["tools"]["summarize"]["model_name"]
        self.generation_kwargs = {
            "temperature": 0.7
        }
    
    async def aclose(self):
        if hasattr(self, "client") and self.client is not None:
            try:
                await self.client.aclose()
            except Exception:
                pass

    def close(self):
        try:
            asyncio.run(self.aclose())
        except RuntimeError:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.aclose())
            except Exception:
                pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
    
    async def _call_llm(self, messages, **override_kwargs):
        generation_kwargs = {
            **self.generation_kwargs,
            **override_kwargs
        }
        
        async def llm_call():
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **generation_kwargs
            )
            return response.choices[0].message.content
        
        return await execute_with_retries(
            llm_call,
            max_attempts=3,
            delay=1.0,
            backoff_factor=2.0,
            context=f"LLM call for summarization"
        )

    async def summarize(self, url: str, raw_content: str, goal: str) -> str:
        async def summarize_operation():
            nonlocal raw_content

            final_summary = None
            if raw_content and not raw_content.startswith("[visit] Failed to read page.") and not raw_content.startswith('[visit] Empty content') and not raw_content.startswith("[document_parser]"):
                raw_content = truncate_to_tokens(raw_content, max_tokens=50000)
                messages = [
                    {"role": "user", "content": EXTRACTOR_PROMPT.format(
                        webpage_content=raw_content, goal=goal
                    )}
                ]
                raw_summarized = await self._call_llm(messages)
                summary_retries = 3
                while len(raw_summarized) < 10 and summary_retries >= 0:
                    truncate_length = int(0.7 * len(raw_content)) if summary_retries > 0 else 25000
                    status_msg = (
                        f"[visit] Summary url[{url}] " 
                        f"attempt {3 - summary_retries + 1}/3, "
                        f"content length: {len(raw_content)}, "
                        f"truncating to {truncate_length} chars"
                    ) if summary_retries > 0 else (
                        f"[visit] Summary url[{url}] failed after 3 attempts, "
                        f"final truncation to 25000 chars"
                    )
                    logger.info(status_msg)
                    raw_content = raw_content[:truncate_length]
                    extraction_prompt = EXTRACTOR_PROMPT.format(
                        webpage_content=raw_content,
                        goal=goal
                    )
                    messages = [{"role": "user", "content": extraction_prompt}]
                    raw_summarized = await self._call_llm(messages)
                    summary_retries -= 1
                
                parse_retry_times = 0
                if isinstance(raw_summarized, str):
                    raw_summarized = raw_summarized.replace("```json", "").replace("```", "").strip()
                
                while parse_retry_times < 2:
                    try:
                        raw_summarized = json.loads(raw_summarized)
                        break
                    except:
                        raw_summarized = await self._call_llm(messages, temperature=1.0)
                        parse_retry_times += 1

                if parse_retry_times >= 2:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Evidence in page: \n" + str(raw_summarized["evidence"]) + "\n\n"
                    useful_information += "Summary: \n" + str(raw_summarized["summary"]) + "\n\n"

                final_summary = useful_information

            else:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                final_summary = useful_information

            return final_summary
        
        return await execute_with_retries(
            summarize_operation,
            max_attempts=5,
            delay=2.0,
            backoff_factor=2.0,
            context=f"Summarization for URL: {url}"
        )


class BriefLLMSummarizer(LLMSummarizer):
    """
    Summarize web content using brief summarize prompt.
    The returned summary only contains goal related summary without the full original content.
    """
    name = "brief_llm_summarizer"
    category = "summarizer"
    def __init__(self, config_path):
        super().__init__(config_path)
    
    async def summarize(self, url: str, raw_content: str, goal: str) -> str:
        async def summarize_operation():
            nonlocal raw_content

            final_summary = None
            if raw_content and not raw_content.startswith("[visit] Failed to read page.") and not raw_content.startswith('[visit] Empty content') and not raw_content.startswith("[document_parser]"):
                raw_content = truncate_to_tokens(raw_content, max_tokens=50000)
                messages = [
                    {"role": "user", "content": EXTRACTOR_PROMPT_BRIEF.format(
                        webpage_content=raw_content, goal=goal
                    )}
                ]
                raw_summarized = await self._call_llm(messages)
                summary_retries = 3
                while len(raw_summarized) < 10 and summary_retries >= 0:
                    truncate_length = int(0.7 * len(raw_content)) if summary_retries > 0 else 25000
                    status_msg = (
                        f"[visit] Summary url[{url}] " 
                        f"attempt {3 - summary_retries + 1}/3, "
                        f"content length: {len(raw_content)}, "
                        f"truncating to {truncate_length} chars"
                    ) if summary_retries > 0 else (
                        f"[visit] Summary url[{url}] failed after 3 attempts, "
                        f"final truncation to 25000 chars"
                    )
                    logger.info(status_msg)
                    raw_content = raw_content[:truncate_length]
                    extraction_prompt = EXTRACTOR_PROMPT_BRIEF.format(
                        webpage_content=raw_content,
                        goal=goal
                    )
                    messages = [{"role": "user", "content": extraction_prompt}]
                    raw_summarized = await self._call_llm(messages)
                    summary_retries -= 1
                
                parse_retry_times = 0
                if isinstance(raw_summarized, str):
                    raw_summarized = raw_summarized.replace("```json", "").replace("```", "").strip()
                
                while parse_retry_times < 2:
                    try:
                        raw_summarized = json.loads(raw_summarized)
                        break
                    except:
                        raw_summarized = await self._call_llm(messages, temperature=1.0)
                        parse_retry_times += 1

                if parse_retry_times >= 2:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    useful_information += "Summary: \n" + str(raw_summarized["summary"]) + "\n\n"

                final_summary = useful_information

            else:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                final_summary = useful_information

            return final_summary
        
        return await execute_with_retries(
            summarize_operation,
            max_attempts=5,
            delay=2.0,
            backoff_factor=2.0,
            context=f"Summarization for URL: {url}"
        )
