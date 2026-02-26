"""
DeepSeek V32 Agent
"""

import re
import os
import sys
import json
import math
import json_repair
import time
import asyncio
import traceback
from copy import deepcopy
from functools import partial
from datetime import datetime
from transformers import AutoTokenizer

if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3])
    sys.path.append(base_path)

from src.agents.base import AgentBase, log_exception_details
from tools.types import SerpEntry
from src.eval import get_judge
from tools.code_sandbox import run_python
from tools.scholar import GoogleScholarTool
from tools.map import GoogleMapsTool
from utils.file_utils import FileUtils
from utils.logger import get_logger
from utils.encoding_dsv32 import encode_messages, parse_message_from_completion_text
from src.llms_async import (
    async_completion,
)

# Helper function to create encode_messages_dsv32
encode_messages_dsv32 = partial(encode_messages, thinking_mode="thinking", context=None, drop_thinking=True, add_default_bos_token=True)

from prompt.system_prompt import (
    SYSTEM_PROMPT_DEEPSEEK_V32,
    TOOLS_DEEPSEEK_V32,
)

logger = get_logger(__name__)


class DeepSeekV32ThinkingWithToolAgent(AgentBase):
    def __init__(self, config_path, **kwargs):
        super().__init__(config_path, **kwargs)
        self.tools = TOOLS_DEEPSEEK_V32
        self._init_special_tokens()
        self._init_tokenizer()
        # Initialize Google Scholar and Google Maps tools
        self.google_scholar_tool = GoogleScholarTool(config_path=config_path, **kwargs)
        self.google_maps_tool = GoogleMapsTool(config_path=config_path, **kwargs)

    def _init_special_tokens(self):
        self.dsml_token = "｜DSML｜"
        self.eos_token = "<｜end▁of▁sentence｜>"
        self.special_token_map = {
            "<function_calls>": "<{dsml_token}function_calls>".format(dsml_token=self.dsml_token),
            "</function_calls>": "</{dsml_token}function_calls>".format(dsml_token=self.dsml_token),
            "<invoke name=": "<{dsml_token}invoke name=".format(dsml_token=self.dsml_token),
            "</invoke>": "</{dsml_token}invoke>".format(dsml_token=self.dsml_token),
            "<parameter name=": "<{dsml_token}parameter name=".format(dsml_token=self.dsml_token),
            "</parameter>": "</{dsml_token}parameter>".format(dsml_token=self.dsml_token),
        }

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2")

    def _format_serp_entry(self, serp_entry: SerpEntry) -> str:
        """Format search result entry based on search tool type"""
        return self._format_serpdev_serp_entry(serp_entry)

    def _format_serpdev_serp_entry(self, serp_entry: SerpEntry) -> str:
        """Format Serper.dev search results"""
        from tools.search import convert_date
        raw_organic = serp_entry.raw_organic or {}
        idx = serp_entry.rank
        link = serp_entry.url
        title = serp_entry.title
        snippet = serp_entry.snippet
        date_published = raw_organic.get("date", "")
        source = raw_organic.get("source", "")

        if date_published != "":
            date_published = convert_date(date_published)
            date_published = f"\nDate published: {date_published}"
        if source != "":
            source = f"\nSource: {source}"
        if snippet != "":
            snippet = f"\n{snippet}"
        formatted_serp = f"{idx}. [{title}]({link}){date_published}{source}\n{snippet}"
        formatted_serp = formatted_serp.replace("Your browser can't play this video.", "")
        return formatted_serp

    def _count_tokens(self, messages: list) -> int:
        encoded_messages = encode_messages_dsv32(messages)
        tokenized_messages = self.tokenizer.encode(encoded_messages)
        return len(tokenized_messages)

    # DeepSeek V3.2 sometimes outputs without DSML tool calls during sampling. Need to resample or manually fix.
    def _fix_completion_text(self, completion_text: str) -> str:
        if not completion_text.endswith("function_calls>") or not completion_text.endswith(self.eos_token):
            completion_text += self.eos_token

        for token, replacement in self.special_token_map.items():
            completion_text = completion_text.replace(token, replacement)
        return completion_text

    async def process_query(self, idx: int, user_query: str, ground_truth: str = None, 
                            dataset_id: str = None, sample_idx: int = 0, override: bool = False):

        if sample_idx > 0:
            if dataset_id:
                save_path = f"{self.config['exps']['output_dir']}/{dataset_id}/problem_{idx}_sample_{sample_idx}.json"
            else:
                save_path = f"{self.config['exps']['output_dir']}/problem_{idx}_sample_{sample_idx}.json"
        else:
            if dataset_id:
                save_path = f"{self.config['exps']['output_dir']}/{dataset_id}/problem_{idx}.json"
            else:
                save_path = f"{self.config['exps']['output_dir']}/problem_{idx}.json"
        
        # Main Agent Loop
        try:
            # Check if there are saved results or checkpoint
            start_iteration = 0
            resume_from_checkpoint = False
            
            # debug flag
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    saved_results = json.load(f)
                
                # Check if this is a checkpoint to resume from
                if saved_results.get("resume", False):
                    # Resume from checkpoint
                    logger.info(f"Problem {idx} sample {sample_idx} found checkpoint at iteration {saved_results.get('current_iteration', 0)}. Resuming...")
                    resume_from_checkpoint = True
                    
                    # Restore messages
                    messages = saved_results["messages"]
                    
                    # Restore time_stats
                    time_stats = {
                        "assistant": saved_results["time_summary"]["assistant"]["details"].copy(),
                        "search": saved_results["time_summary"]["search"]["details"].copy(),
                        "visit": saved_results["time_summary"]["visit"]["details"].copy(),
                        "summarize": saved_results["time_summary"]["summarize"]["details"].copy(),
                        "python": saved_results["time_summary"]["python"]["details"].copy(),
                        "google_scholar": saved_results["time_summary"].get("google_scholar", {}).get("details", []).copy(),
                        "google_maps": saved_results["time_summary"].get("google_maps", {}).get("details", []).copy()
                    }
                    
                    # Restore iteration (start from next iteration)
                    start_iteration = saved_results.get("current_iteration", 0) + 1
                    
                    logger.info(f"Problem {idx} sample {sample_idx} resumed from iteration {start_iteration}")
                else:
                    # Completed results
                    if override:
                        logger.info(f"Problem {idx} sample {sample_idx} already solved. Overriding...")
                    else:
                        logger.info(f"Problem {idx} sample {sample_idx} already solved. Skipping...")
                        return saved_results
            
            # system prompt for debugging
            #system_prompt = "Please use the following tools (search, visit, and python) to answer the question. "
            # Initialize if not resuming from checkpoint
            if not resume_from_checkpoint:
                logger.info(f"Processing query {idx} started")
                # messages = [
                #     {"role": "system", "content": SYSTEM_PROMPT_GPT_OSS_V22},
                #     #{"role": "system", "content": system_prompt},
                #     {"role": "developer", "content": user_query, "tools": TOOLS_DEEPSEEK_V32}
                # ]
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_DEEPSEEK_V32, "tools": TOOLS_DEEPSEEK_V32},
                    {"role": "user", "content": user_query}
                ]

                # Initialize time statistics
                time_stats = {
                    "assistant": [],
                    "search": [],
                    "visit": [],
                    "summarize": [],
                    "python": [],
                    "google_scholar": [],
                    "google_maps": []
                }

            force_answer = False
            termination = None
            # Initialize AssertionError counter
            assertion_error_count = 0
            max_assertion_errors = 25 # default threshold is 5
            for iteration in range(start_iteration, self.config["agent"]["max_iter"]):
                logger.info("Query %s, Iteration %s started, max iteration is %s", idx, iteration, self.config["agent"]["max_iter"])
                # Model generation step
                try:
                    # Need to manually use dsv32 encode messages

                    encoded_messages = encode_messages_dsv32(messages)
                    start_time = time.time()
                    response = await async_completion(encoded_messages, self.config["llm"]["local"]["model_name"], self.llm_endpoint)
                    response_text = self._fix_completion_text(response.text)
                    response_msg = parse_message_from_completion_text(response_text, thinking_mode="thinking")
                    #print(response_msg["reasoning_content"])
                    assistant_elapsed_time = time.time() - start_time
                    time_stats["assistant"].append(assistant_elapsed_time)
                except AssertionError as e:
                    assertion_error_count += 1
                    logger.warning(f"Query {idx}, Iteration {iteration}: Failed to parse messages from completion text. Skipping this iteration. (AssertionError count: {assertion_error_count}/{max_assertion_errors})")
                    if assertion_error_count >= max_assertion_errors:
                        logger.error(f"Query {idx}: AssertionError count ({assertion_error_count}) reached threshold ({max_assertion_errors}). Breaking the iteration loop.")
                        termination = "infinity loop"
                        messages.append({
                            "role": "assistant",
                            "content": response_text
                        })
                        break
                    continue
                except Exception as e:
                    log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: LLM call failed - . Skipping this iteration.")
                    continue

                # Parse tool calls
                if len(response_msg.get("tool_calls", [])) > 0:
                    # Record messages length before adding assistant_msg for rollback on parse failure
                    msg_len_before_assistant = len(messages)
                    assistant_msg = response_msg
                    assistant_msg.update({
                        "time": {
                            "name": "assistant",
                            "elapsed_time": assistant_elapsed_time
                        }
                    })
                    messages.append(assistant_msg)
                    
                    # Mark whether to skip this iteration
                    should_skip_iteration = False
                    
                    for tool_call in response_msg.get("tool_calls", []):
                        tool_name = tool_call.get("function", {}).get("name")
                        raw_tool_args = tool_call.get("function", {}).get("arguments")

                        # Handle python tool call
                        if tool_name == "python":
                            # Parse python code
                            try:
                                code = json.loads(raw_tool_args)["code"]
                                code_lines = [line for line in code.split('\n') if line.strip()]
                                last_line = code_lines[-1]
                                if 'print' not in last_line:
                                    last_line = f"print({last_line})"
                                code_lines[-1] = last_line
                                code = "\n".join(code_lines)
                            except Exception as e:
                                log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to parse tool call arguments for tool {tool_name}")
                                logger.warning(f"Query {idx}, Iteration {iteration}: Failed to parse [python] tool call arguments. Skip this iteration.")
                                # Rollback to state before assistant_msg
                                messages = messages[:msg_len_before_assistant]
                                should_skip_iteration = True
                                break
                            # Execute python code if parsing succeeds
                            try:
                                python_start_time = time.time()
                                result = await run_python(code, **self.config["tools"]["python"])
                                python_elapsed_time = time.time() - python_start_time
                                time_stats["python"].append(python_elapsed_time)
                                messages.append({
                                    "role": "tool",
                                    "id": None,
                                    "content": result,
                                    "time": {
                                        "name": "python",
                                        "elapsed_time": python_elapsed_time
                                    }
                                })
                            except Exception as e:
                                log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to execute python for tool {tool_name}")
                                messages.append({
                                    "role": "tool",
                                    "id": None,
                                    "content": json.dumps("Failed to execute python."),
                                    "time": {
                                        "name": "python",
                                        "elapsed_time": 0.0
                                    }
                                })
                                continue
                        
                        # Handle search tool call
                        if tool_name == "search":
                            # Parse search parameters
                            try:
                                func_args = json_repair.loads(raw_tool_args)
                                queries = func_args["query"] # list of str
                                if isinstance(queries, str):
                                    queries = [queries]
                            except Exception as e:
                                log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to parse tool call arguments for tool {tool_name}")
                                logger.warning(f"Query {idx}, Iteration {iteration}: Failed to parse [search] tool call arguments. Skip this iteration.")
                                # Rollback to state before assistant_msg
                                messages = messages[:msg_len_before_assistant]
                                should_skip_iteration = True
                                break
                            
                            # Execute search
                            try:
                                search_start_time = time.time()
                                search_results_list = await asyncio.gather(
                                    *(self.search_tool.search(search_query) for search_query in queries)
                                )
                                search_elapsed_time = time.time() - search_start_time
                                time_stats["search"].append(search_elapsed_time)
                                snippets_for_all_queries = []
                                for search_results in search_results_list:
                                    search_query = search_results.query
                                    snippets_for_this_query = []
                                    # Use FAILED_TO_SEARCH as a special marker to indicate search error
                                    if search_results.query.startswith("Failed to search for"):
                                        snippets_for_all_queries.append(search_results.query)
                                        continue
                                    for serp_entry in search_results.results:
                                        formatted_serp = self._format_serp_entry(serp_entry)
                                        snippets_for_this_query.append(formatted_serp)
                                    snippets_for_this_query = f"A Google search for '{search_query}' found {len(snippets_for_this_query)} results:\n\n## Web Results\n" + "\n\n".join(snippets_for_this_query)
                                    snippets_for_all_queries.append(snippets_for_this_query)
                                snippets_for_all_queries = "\n=======\n".join(snippets_for_all_queries)
                                messages.append({
                                    "role": "tool",
                                    "id": None,
                                    "content": snippets_for_all_queries,
                                    "time": {
                                        "name": "search",
                                        "elapsed_time": search_elapsed_time
                                    }
                                })
                            except Exception as e:
                                log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to execute search for tool {tool_name}")
                                messages.append({
                                    "role": "tool",
                                    "id": None,
                                    "content": "Failed to execute search. Please check the search queries (list[str]) and try again.",
                                    "time": {
                                        "name": "search",
                                        "elapsed_time": 0.0
                                    }
                                })
                                continue

                        # Handle web visit tool call
                        if tool_name == "visit":
                            try:
                                func_args = json_repair.loads(raw_tool_args)
                                goal = func_args["goal"] # str
                                urls = func_args["url"] # supported to be list of str
                                # case1: urls from model is not list type but wrapped in a string. Need to json.loads again (DeepSeek often has this issue)
                                if isinstance(urls, str) and urls.strip().startswith("[") and urls.strip().endswith("]"):
                                    urls = json.loads(urls.strip())
                                # case2: urls from model is str, convert to list[str]
                                elif isinstance(urls, str):
                                    urls = [urls]
                                # case3: urls from model is list[str], use directly
                                else:
                                    pass

                            except Exception as e:
                                log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to parse tool call arguments for tool {tool_name}")
                                logger.warning(f"Query {idx}, Iteration {iteration}: Failed to parse [visit] tool call arguments. Skip this iteration.")
                                # Rollback to state before assistant_msg
                                messages = messages[:msg_len_before_assistant]
                                should_skip_iteration = True
                                break

                            try:
                                # Timing: visit (fetch web content)
                                visit_start_time = time.time()
                                web_contents = await asyncio.gather(
                                    *(self.web_crawl_tool.crawl(url) for url in urls)
                                )


                                if self.web_crawl_tool_backup is not None:
                                    for idx_web, (url, web_content) in enumerate(zip(urls, web_contents)):
                                        if web_content.text_content is None:
                                            try:
                                                logger.info("Crawling page %s using backup tool", url)
                                                web_content_backup = await self.web_crawl_tool_backup.crawl(url)
                                                if web_content_backup.text_content is not None:
                                                    web_contents[idx_web] = web_content_backup
                                                else:
                                                    logger.error("Failed to crawl page %s using backup tool", url)
                                            except Exception as e:
                                                logger.error("Failed to crawl page %s using backup tool: %s", url, e)
                                                logger.error(traceback.format_exc())
                                
                                visit_elapsed_time = time.time() - visit_start_time
                                time_stats["visit"].append(visit_elapsed_time)
                                
                                # Timing: summarize (call LLM for summary)
                                summarize_start_time = time.time()
                                summarized_contents = await asyncio.gather(
                                    *(self.web_summarize_tool.summarize(
                                        url=url,
                                        raw_content=web_content.text_content,
                                        goal=goal
                                    ) for url, web_content in zip(urls, web_contents))
                                )

                                for web_content_idx, web_content in enumerate(web_contents):
                                    if web_content.text_content is None:
                                        summarized_contents[web_content_idx] = "The web page is not accessible."
                                summarize_elapsed_time = time.time() - summarize_start_time
                                time_stats["summarize"].append(summarize_elapsed_time)
                                
                                result = "\n=======\n".join(summarized_contents)
                                messages.append({
                                    "role": "tool",
                                    "id": None,
                                    "content": result,
                                    "time": [
                                        {
                                            "name": "visit",
                                            "elapsed_time": visit_elapsed_time
                                        },
                                        {
                                            "name": "summarize",
                                            "elapsed_time": summarize_elapsed_time
                                        }
                                    ]
                                })
                                        
                            except Exception as e:
                                log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to execute web visit for tool {tool_name}")
                                messages.append({
                                    "role": "tool",
                                    "id": None,
                                    "content": "Web visit failed. The access to the URL is blocked or denied.",
                                })
                                continue
                        
                        # Handle google_scholar tool call
                        if tool_name == "google_scholar":
                            try:
                                func_args = json_repair.loads(raw_tool_args)
                                queries = func_args["query"]  # list of str
                                if isinstance(queries, str):
                                    queries = [queries]
                            except Exception as e:
                                log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to parse tool call arguments for tool {tool_name}")
                                logger.warning(f"Query {idx}, Iteration {iteration}: Failed to parse [google_scholar] tool call arguments. Skip this iteration.")
                                messages = messages[:msg_len_before_assistant]
                                should_skip_iteration = True
                                break
                            
                            try:
                                scholar_start_time = time.time()
                                scholar_results_list = await asyncio.gather(
                                    *(self.google_scholar_tool.search(query) for query in queries)
                                )
                                scholar_elapsed_time = time.time() - scholar_start_time
                                time_stats["google_scholar"].append(scholar_elapsed_time)
                                
                                formatted_results = []
                                for scholar_results in scholar_results_list:
                                    formatted_results.append(self.google_scholar_tool.format_results(scholar_results))
                                result = "\n=======\n".join(formatted_results)
                                
                                messages.append({
                                    "role": "tool",
                                    "id": None,
                                    "content": result,
                                    "time": {
                                        "name": "google_scholar",
                                        "elapsed_time": scholar_elapsed_time
                                    }
                                })
                            except Exception as e:
                                log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to execute google_scholar for tool {tool_name}")
                                messages.append({
                                    "role": "tool",
                                    "id": None,
                                    "content": "Failed to execute Google Scholar search. Please check the query and try again.",
                                    "time": {
                                        "name": "google_scholar",
                                        "elapsed_time": 0.0
                                    }
                                })
                                continue
                        
                        # Handle google_maps tool call
                        if tool_name == "google_maps":
                            try:
                                func_args = json_repair.loads(raw_tool_args)
                                query = func_args["q"]  # str
                                page = func_args.get("page", 1)  # int, default 1
                            except Exception as e:
                                log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to parse tool call arguments for tool {tool_name}")
                                logger.warning(f"Query {idx}, Iteration {iteration}: Failed to parse [google_maps] tool call arguments. Skip this iteration.")
                                messages = messages[:msg_len_before_assistant]
                                should_skip_iteration = True
                                break
                            
                            try:
                                maps_start_time = time.time()
                                maps_results = await self.google_maps_tool.search(query, page=page)
                                maps_elapsed_time = time.time() - maps_start_time
                                time_stats["google_maps"].append(maps_elapsed_time)
                                
                                result = self.google_maps_tool.format_results(maps_results)
                                
                                messages.append({
                                    "role": "tool",
                                    "id": None,
                                    "content": result,
                                    "time": {
                                        "name": "google_maps",
                                        "elapsed_time": maps_elapsed_time
                                    }
                                })
                            except Exception as e:
                                log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to execute google_maps for tool {tool_name}")
                                messages.append({
                                    "role": "tool",
                                    "id": None,
                                    "content": "Failed to execute Google Maps search. Please check the query and try again.",
                                    "time": {
                                        "name": "google_maps",
                                        "elapsed_time": 0.0
                                    }
                                })
                                continue
                    
                    # Skip this iteration if tool call parsing fails
                    if should_skip_iteration:
                        continue

                # This round model only has reasoning, no tool calls   
                # case1: found answer in final round
                # case2: parse error?
                else:
                    # case1: found answer, stop agent loop directly
                    if response_msg.get("content") is not None and "<answer>" in response_msg.get("content"):
                        answer_content = ""
                        if response_msg.get("reasoning_content"):
                            answer_content += response_msg.get("reasoning_content")
                        if response_msg.get("content"):
                            answer_content += response_msg.get("content")
                        assistant_msg = {
                            "role": "assistant",
                            "content": answer_content,
                            "msg_reasoning": response_msg.get("reasoning_content"),
                            "msg_content": response_msg.get("content"),
                            "time": {
                                "name": "assistant",
                                "elapsed_time": assistant_elapsed_time
                            }
                        }
                        messages.append(assistant_msg)
                        termination = "answer"
                        break

                    # case2: possible parse error, no answer found. Skip this iteration
                    else:
                        logger.warning(f"Query {idx}, Iteration {iteration}: Failed to find tool_calls and answer in an generation round. Skip this iteration.")

                        continue

                    
                # Consider context length and max iteration reached | need to force termination and answer
                if iteration == self.config['agent']['max_iter'] - 1:
                    termination = "iteration"
                    force_answer = True
                    logger.info(f"Query {idx}: The number of iterations reached the maximum limit: {self.config['agent']['max_iter']}. Force answering the question.")
                
                token_count = self._count_tokens(messages)
                max_tokens = 120 * 1024 # 128k max length, leave 4k for generation
                if token_count > max_tokens:
                    termination = "context length"
                    force_answer = True
                    logger.info(f"Query {idx}: The context length reached the maximum limit: {token_count} tokens. Rollback messages to fit and force answering the question.")
                else:
                    force_answer = False               
                
                if force_answer:
                    # Rollback to find messages with length under limit and ending with assistant or tool role
                    truncated_messages = []
                    for msg in messages:
                        # Try to add current message
                        truncated_messages.append(msg)
                        # Calculate token count for entire message list
                        current_token_count = self._count_tokens(truncated_messages)
                        if current_token_count > max_tokens:
                            # Remove just added message and stop if exceeding limit
                            truncated_messages.pop()
                            break

                    # Find truncation point ending with assistant or tool role from end to start
                    for i in range(len(truncated_messages)-1, -1, -1):
                        if truncated_messages[i]["role"] in {"assistant"}:
                            truncated_messages = truncated_messages[:i+1]
                            break

                    messages = truncated_messages
                    messages.append({
                        "role": "tool", 
                        "content": "The context length reached the maximum context length / iteration limit. Please answer the question based on previous context. The answer should be wrapped in <answer> </answer> tags."
                    })
                
                    # Filter time field before sending to LLM
                    encoded_messages = encode_messages_dsv32(messages)
                    force_answer_start_time = time.time()
                    response = await async_completion(encoded_messages, self.config["llm"]["local"]["model_name"], self.llm_endpoint)
                    response_text = self._fix_completion_text(response.text)
                    response_msg = parse_message_from_completion_text(response_text, thinking_mode="thinking")
                    force_answer_elapsed_time = time.time() - force_answer_start_time
                    time_stats["assistant"].append(force_answer_elapsed_time)

                    answer_content = ""
                    
                    if response_msg.get("reasoning_content"):
                        answer_content += response_msg.get("reasoning_content")
                    if response_msg.get("content"):
                        answer_content += response_msg.get("content")
                    messages.append({
                        "role": "assistant",
                        "content": answer_content,
                        "msg_reasoning": response_msg.get("reasoning_content"),
                        "msg_content": response_msg.get("content"),
                        "time": {
                            "name": "assistant",
                            "elapsed_time": force_answer_elapsed_time
                        }
                    })

                    break

                # Save checkpoint every K steps for breakpoint recovery
                checkpoint_interval = 10  # save checkpoint every 10 steps
                if (iteration + 1) % checkpoint_interval == 0:
                    logger.info(f"Query {idx}: Saving checkpoint at iteration {iteration + 1}")
                    # Calculate current time summary
                    checkpoint_time_summary = {
                        "assistant": {
                            "count": len(time_stats["assistant"]),
                            "total_time": sum(time_stats["assistant"]),
                            "details": time_stats["assistant"]
                        },
                        "search": {
                            "count": len(time_stats["search"]),
                            "total_time": sum(time_stats["search"]),
                            "details": time_stats["search"]
                        },
                        "visit": {
                            "count": len(time_stats["visit"]),
                            "total_time": sum(time_stats["visit"]),
                            "details": time_stats["visit"]
                        },
                        "summarize": {
                            "count": len(time_stats["summarize"]),
                            "total_time": sum(time_stats["summarize"]),
                            "details": time_stats["summarize"]
                        },
                        "python": {
                            "count": len(time_stats["python"]),
                            "total_time": sum(time_stats["python"]),
                            "details": time_stats["python"]
                        },
                        "google_scholar": {
                            "count": len(time_stats["google_scholar"]),
                            "total_time": sum(time_stats["google_scholar"]),
                            "details": time_stats["google_scholar"]
                        },
                        "google_maps": {
                            "count": len(time_stats["google_maps"]),
                            "total_time": sum(time_stats["google_maps"]),
                            "details": time_stats["google_maps"]
                        },
                        "total_time": sum(time_stats["assistant"]) + sum(time_stats["search"]) + sum(time_stats["visit"]) + sum(time_stats["summarize"]) + sum(time_stats["python"]) + sum(time_stats["google_scholar"]) + sum(time_stats["google_maps"])
                    }
                    checkpoint_result = {
                        "idx": idx,
                        "dataset_id": dataset_id,
                        "question": user_query,
                        "answer": ground_truth,
                        "messages": messages,
                        "prediction": None,
                        "termination": None,
                        "processing_time_seconds": checkpoint_time_summary["total_time"],
                        "time_summary": checkpoint_time_summary,
                        "llm_as_judge": None,
                        "resume": True,
                        "current_iteration": iteration
                    }
                    with open(save_path, "w") as f:
                        json.dump(checkpoint_result, f, indent=4)

            
            # agent loop ends.
            prediction = messages[-1].get("msg_content", messages[-1]["content"])
            # Calculate time summary
            time_summary = {
                "assistant": {
                    "count": len(time_stats["assistant"]),
                    "total_time": sum(time_stats["assistant"]),
                    "details": time_stats["assistant"]
                },
                "search": {
                    "count": len(time_stats["search"]),
                    "total_time": sum(time_stats["search"]),
                    "details": time_stats["search"]
                },
                "visit": {
                    "count": len(time_stats["visit"]),
                    "total_time": sum(time_stats["visit"]),
                    "details": time_stats["visit"]
                },
                "summarize": {
                    "count": len(time_stats["summarize"]),
                    "total_time": sum(time_stats["summarize"]),
                    "details": time_stats["summarize"]
                },
                "python": {
                    "count": len(time_stats["python"]),
                    "total_time": sum(time_stats["python"]),
                    "details": time_stats["python"]
                },
                "google_scholar": {
                    "count": len(time_stats["google_scholar"]),
                    "total_time": sum(time_stats["google_scholar"]),
                    "details": time_stats["google_scholar"]
                },
                "google_maps": {
                    "count": len(time_stats["google_maps"]),
                    "total_time": sum(time_stats["google_maps"]),
                    "details": time_stats["google_maps"]
                },
                "total_time": sum(time_stats["assistant"]) + sum(time_stats["search"]) + sum(time_stats["visit"]) + sum(time_stats["summarize"]) + sum(time_stats["python"]) + sum(time_stats["google_scholar"]) + sum(time_stats["google_maps"])
            }
            
            # Get judge and evaluate
            remote = self.config.get("judge", {}).get("remote", True)
            model_name = self.config.get("judge", {}).get("local", {}).get("model_name")
            endpoint = self.config.get("judge", {}).get("local", {}).get("endpoint")
            judger = get_judge(dataset_id, remote, model_name, endpoint)
            judge_result = judger.judge(user_query, prediction, ground_truth)
            # count tokens consumption
            try:
                enc_msg_prev = encode_messages(messages[:-1], thinking_mode="thinking", context=None, drop_thinking=True, add_default_bos_token=False)
                enc_msg_curr = encode_messages(messages[-1:], thinking_mode="chat", context=None, drop_thinking=True, add_default_bos_token=False)
                tokens_count = len(self.tokenizer.encode(enc_msg_prev)) + len(self.tokenizer.encode(enc_msg_curr))
            except Exception as e:
                tokens_count = None
            result = {
                "idx": idx,
                "dataset_id": dataset_id,
                "question": user_query,
                "answer": ground_truth,
                "messages": messages,
                "prediction": prediction,
                "termination": termination,
                "processing_time_seconds": time_summary["total_time"],
                "time_summary": time_summary,
                "llm_as_judge": judge_result,
                "turns": math.ceil(max(0, len(messages) - 2) / 2),
                "tokens_count": tokens_count
            }
            with open(save_path, "w") as f:
                json.dump(result, f, indent=4)
            return result
        except Exception as e:
            logger.error(f"Query {idx}: Failed to process query: {e}")
            log_exception_details(logger, e, f"Query {idx}: Failed to process query")
            raise e


