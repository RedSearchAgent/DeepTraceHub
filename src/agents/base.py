"""
AgentBase: Base class for agents.
"""

import re
import os
import sys
import random
import asyncio
import aiohttp
import traceback
import threading
from typing import List, Dict

if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3])
    sys.path.append(base_path)

from tools.search import (
    EmptyWebSearchTool,
    SerperDevSearchTool,
    LocalWebSearchTool,
)

from tools.summarizer import (
    LLMSummarizer,
    BriefLLMSummarizer,
)
from tools.search import (
    JinaWebCrawlTool,
    SerperDevWebCrawlTool,
    LocalWebCrawlTool,
)
from utils.file_utils import FileUtils
from utils.logger import get_logger

from prompt.system_prompt import (
    SYSTEM_PROMPT_XHS_V2
)

logger = get_logger(__name__)

TOOL_CLASS_SEARCH = [
    EmptyWebSearchTool,
    SerperDevSearchTool,
    LocalWebSearchTool,
]
TOOL_CLASS_CRAWL = [
    JinaWebCrawlTool,
    SerperDevWebCrawlTool,
    LocalWebCrawlTool,
]
TOOL_CLASS_SUMMARIZER = [
    LLMSummarizer,
    BriefLLMSummarizer,
]
TOOL_MAP_SEARCH = {item.name: item for item in TOOL_CLASS_SEARCH}
TOOL_MAP_CRAWL = {item.name: item for item in TOOL_CLASS_CRAWL}
TOOL_MAP_SUMMARIZER = {item.name: item for item in TOOL_CLASS_SUMMARIZER}

MODEL_ARGS = {
    "redsearcher": {
        "tokenizer": "Qwen/Qwen3-30B-A3B",
        "system_prompt": SYSTEM_PROMPT_XHS_V2
    }
}

def log_exception_details(logger, exception, context=""):
    """Record detailed exception information"""

    logger.error(f"{context}Exception occurred:")
    logger.error(f"  Type: {type(exception).__name__}")
    logger.error(f"  Message: {str(exception)}")
    logger.error(f"  Traceback:\n{traceback.format_exc()}")


class AgentBase:
    def __init__(self, config_path="config/config.yaml", **kwargs):
        self.config = FileUtils.load_config(config_path)
        self.search_tool = TOOL_MAP_SEARCH[self.config["tools"]["search_tool"]](config_path,**kwargs)
        self.web_crawl_tool = TOOL_MAP_CRAWL[self.config["tools"]["crawl_tool"]](config_path,**kwargs)
        self.web_summarize_tool = TOOL_MAP_SUMMARIZER.get(self.config["tools"]["summarize_tool"], LLMSummarizer)(config_path)
        if self.config["tools"]["crawl_tool_backup"] is not None:
            logger.info("Enabled backup crawl tool: %s", self.config["tools"]["crawl_tool_backup"])
            self.web_crawl_tool_backup = TOOL_MAP_CRAWL[self.config["tools"]["crawl_tool_backup"]](config_path,**kwargs)
        else:
            self.web_crawl_tool_backup = None

        self._init_llm_endpoint_load_balance()
    
    def _init_llm_endpoint_load_balance(self):
        # Compared to random load balancing in _init_llm_endpoint, this function achieves load balancing by accessing the endpoints list.
        # sgl load balancing based on: curl -s http://10.144.172.80:18901/metrics | egrep '^sglang:num_(running|queue)_reqs
        # vllm load balancing: same as above.
        # http://ip:port/v1 -> http://ip:port/metrics 
        if self.config["llm"]["local"]["endpoint_list"] is None:
            self.llm_endpoint = self.config["llm"]["local"]["endpoint"]
            return

        def _run_coro_sync(coro):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)

            out = {"result": None, "error": None}
            def runner():
                try:
                    out["result"] = asyncio.run(coro)
                except Exception as e:
                    out["error"] = e
            t = threading.Thread(target=runner, daemon=True)
            t.start(); t.join()
            if out["error"] is not None:
                raise out["error"]
            return out["result"]

        async def _check_backend(endpoint_list):
            url = random.choice(endpoint_list) + "/models"
            timeout = aiohttp.ClientTimeout(total=3)  # Adjust as needed
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as resp:
                        resp.raise_for_status()           # Non-2xx raises exception directly
                        res = await resp.json()
                        owned_by = res['data'][0]['owned_by']
            except Exception as e:
                owned_by = None
            return owned_by

        async def _fetch_one(
            session: aiohttp.ClientSession,
            sem: asyncio.Semaphore,
            url: str,
        ):
            url_metrics = url.replace("/v1", "/metrics")
            async with sem:
                try:
                    async with session.get(url_metrics) as resp:
                        resp.raise_for_status()
                        text = await resp.text()
                        result = re.findall(re_pattern, text)
                        result_dict = {}
                        for item in result:
                            result_dict[item[0]] = item[1]
                        return url, result_dict, None
                except Exception as e:
                    print(f"Request {url_metrics} failed: {e}")
                    return url, None, e

        async def _fetch_all(urls: List[str]):
            sem = asyncio.Semaphore(max_concurrency)
            timeout_cfg = aiohttp.ClientTimeout(total=timeout)

            # Limit connections with connector to avoid connection overflow from too many endpoints
            connector = aiohttp.TCPConnector(limit=max_concurrency)
            async with aiohttp.ClientSession(timeout=timeout_cfg, connector=connector) as session:
                tasks = [_fetch_one(session, sem, u) for u in urls]
                results = await asyncio.gather(*tasks, return_exceptions=False)

            results = [item for item in results if item[2] is None]
            return results
        
        def _sort_func_load_balance(item: Dict):
            if back_end == "sglang":
                num_running_reqs, num_queue_reqs, token_usage = float(item["num_running_reqs"]), float(item["num_queue_reqs"]), float(item["token_usage"])
            elif back_end == "vllm":
                num_running_reqs, num_queue_reqs, token_usage = float(item["num_requests_running"]), float(item["num_requests_waiting"]), float(item["gpu_cache_usage_perc"])
            else:
                return random.random()

            SAT_TH = 9.0
            saturated = (num_running_reqs >= SAT_TH)
            token_scale = 25.0 if saturated else 10.0
            adjusted_queue = num_queue_reqs + token_usage * token_scale
            RUN_W = 1000.0
            SEC_W = 25.0 if saturated else 10.0
            # Add very small jitter to prevent mass ties
            jitter = random.random() * 1e-6
            score = (
                num_running_reqs * RUN_W +
                adjusted_queue * SEC_W +
                jitter
            )
            return score

        endpoint_list = self.config["llm"]["local"]["endpoint_list"]
        #back_end = asyncio.run(_check_backend(endpoint_list))
        back_end = _run_coro_sync(_check_backend(endpoint_list))
        if back_end == "sglang":
            re_pattern = r"(?m)^(?:[a-zA-Z_:]*:)?(num_running_reqs|num_queue_reqs|token_usage)(?:\{[^}]*\})?\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
        elif back_end == "vllm":
            re_pattern= r"(?m)^(?:[a-zA-Z_:]*:)?(num_requests_running|num_requests_waiting|gpu_cache_usage_perc)(?:\{[^}]*\})?\s+([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
        else:
            re_pattern = None

        max_concurrency, timeout = 5, 5

        #if back_end in ["sglang", "vllm"]:
        if back_end in ["sglang"]:
            #load_balance_infos = asyncio.run(_fetch_all(endpoint_list))
            load_balance_infos = _run_coro_sync(_fetch_all(endpoint_list))
            sorted_load_balance_infos = sorted(load_balance_infos, key=lambda x: _sort_func_load_balance(x[1]))
            self.llm_endpoint = sorted_load_balance_infos[0][0]
        else:
            self.llm_endpoint = random.choice(endpoint_list)

        
    def _init_llm_endpoint(self):
        # Used for sticky requests to vLLM engine to maximize prefix-caching utilization
        if self.config["llm"]["local"]["endpoint_list"] is not None and len(self.config["llm"]["local"]["endpoint_list"]) > 0:
            self.llm_endpoint = random.choice(self.config["llm"]["local"]["endpoint_list"])
        else:
            self.llm_endpoint = self.config["llm"]["local"]["endpoint"]
            
    async def process_query(self, idx: int, user_query: str, ground_truth: str = None, 
                            dataset_id: str = None, sample_idx: int = 0, override: bool = False):
        """
        处理单个查询的异步方法
        
        Args:
            idx: Query index
            user_query: User query
            ground_truth: Ground truth answer
            dataset_id: Dataset ID
            sample_idx: Sample index
            override: Whether to override existing result files
        """
        raise NotImplementedError("Subclass must implement this method: AgentBase.process_query")

