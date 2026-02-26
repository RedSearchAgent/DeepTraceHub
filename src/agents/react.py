"""
ReACTAgent: Vanilla ReACT Agent
"""

import re
import os
import sys
import json
import json5
import math
import time
import asyncio
import traceback
from datetime import datetime
from transformers import AutoTokenizer

if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3])
    sys.path.append(base_path)

from src.agents.base import AgentBase, MODEL_ARGS, log_exception_details
from tools.types import SerpEntry, WebContent
from tools.search import convert_date
from src.eval import get_judge
from tools.code_sandbox import run_python
from utils.logger import get_logger
from src.llms_async import async_chat_completion

logger = get_logger(__name__)


class ReACTAgent(AgentBase):
    def __init__(self, config_path, **kwargs):
        super().__init__(config_path, **kwargs)
        self.model_args = MODEL_ARGS[self.config["agent"]["model"]]
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_args["tokenizer"])
    
    def _count_tokens(self, messages):
        encoding = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
        return len(encoding)

    def _filter_time_from_messages(self, messages: list) -> list:
        """Filter time field from messages for passing to LLM"""
        filtered_messages = []
        for msg in messages:
            filtered_msg = {k: v for k, v in msg.items() if k != "time"}
            filtered_messages.append(filtered_msg)
        return filtered_messages

    def _format_serp_entry(self, serp_entry: SerpEntry) -> str:
        return self._format_serpdev_serp_entry(serp_entry)
        
    def _format_serpdev_serp_entry(self, serp_entry: SerpEntry) -> str:
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

    def _post_process_code(self, code: str) -> str:
        code_lines = [line for line in code.split("\n") if line.strip()]
        last_line = code_lines[-1]
        if 'print' not in last_line:
            last_line = f"print({last_line})"
        code_lines[-1] = last_line
        code = "\n".join(code_lines)
        return code

    async def _execute_google_maps_tool(
        self, tool_args: dict, tool_name: str, idx: int, iteration: int,
        messages: list, msg_len_before_assistant: int, time_stats: dict
    ) -> bool:
        # Parse search parameters
        try:
            search_query = tool_args.get('q', '')
        except Exception as e:
            log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to parse tool call arguments for tool {tool_name}")
            logger.warning(f"Query {idx}, Iteration {iteration}: Failed to parse [{tool_name}] tool call arguments. Skip this iteration.")
            messages[:] = messages[:msg_len_before_assistant]
            return True  # need to break
        
        # Execute search
        try:
            search_start_time = time.time()
            search_result = await self.google_maps_tool.search(search_query)
            search_elapsed_time = time.time() - search_start_time
            time_stats["google_maps"].append(search_elapsed_time)
            formatted_search_result = self.google_maps_tool.format_results(search_result)
            search_result = f"<tool_response>\n{formatted_search_result}\n</tool_response>"
        except Exception as e:
            log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to execute google maps search for tool {tool_name}")
            search_result = f"<tool_response>\nFailed to execute Google Maps search. Please check the query and try again.\n</tool_response>"
            search_elapsed_time = 0.0
            time_stats["google_maps"].append(search_elapsed_time)
        finally:
            messages.append({
                "role": "tool",
                "content": search_result,
                "time": {
                    "name": "google_maps",
                    "elapsed_time": search_elapsed_time
                }
            })
        return False

    async def _execute_google_scholar_tool(
        self, tool_args: dict, tool_name: str, idx: int, iteration: int,
        messages: list, msg_len_before_assistant: int, time_stats: dict
    ) -> bool:  
        # Parse search parameters
        try:
            search_queries = tool_args.get('query', [])
            if isinstance(search_queries, str):
                search_queries = [search_queries]
        except Exception as e:
            log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to parse tool call arguments for tool {tool_name}")
            logger.warning(f"Query {idx}, Iteration {iteration}: Failed to parse [{tool_name}] tool call arguments. Skip this iteration.")
            messages[:] = messages[:msg_len_before_assistant]
            return True  # need to break

        # Execute search
        try:
            search_start_time = time.time()
            search_result_list = await asyncio.gather(
                *(self.google_scholar_tool.search(search_query) for search_query in search_queries)
            )
            search_elapsed_time = time.time() - search_start_time
            time_stats["google_scholar"].append(search_elapsed_time)
            formatted_search_result_list = [
                self.google_scholar_tool.format_results(search_result) for search_result in search_result_list
            ]
            snippets = "\n=======\n".join(formatted_search_result_list).strip()
            search_result = f"<tool_response>\n{snippets}\n</tool_response>"
        except Exception as e:
            log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to execute google scholar search for tool {tool_name}")
            search_result = f"<tool_response>\nFailed to execute Google Scholar search. Please check the query and try again.\n</tool_response>"
            search_elapsed_time = 0.0
            time_stats["google_scholar"].append(search_elapsed_time)
        finally:
            messages.append({
                "role": "tool",
                "content": search_result,
                "time": {
                    "name": "google_scholar",
                    "elapsed_time": search_elapsed_time
                }
            })
        return False

    async def _execute_search_tool(self, tool_args: dict, tool_name: str, idx: int, iteration: int,
                                     messages: list, msg_len_before_assistant: int, time_stats: dict) -> bool:
        """
        执行搜索工具（通用方法，支持 search 和 text_search_bing）
        
        Returns:
            bool: True means need to break (parameter parsing failed), False means continue
        """
        # Stage 2a: Parse search parameters
        try:
            search_queries = tool_args.get('query', [])
            if not isinstance(search_queries, list):
                search_queries = [search_queries]
        except Exception as e:
            log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to parse tool call arguments for tool {tool_name}")
            logger.warning(f"Query {idx}, Iteration {iteration}: Failed to parse [{tool_name}] tool call arguments. Skip this iteration.")
            messages[:] = messages[:msg_len_before_assistant]
            return True  # need to break
        
        # Stage 2b: Execute search
        try:
            search_start_time = time.time()
            search_results_list = await asyncio.gather(
                *(self.search_tool.search(search_query) for search_query in search_queries)
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
            snippets_for_all_queries = "\n=======\n".join(snippets_for_all_queries).strip()
            search_result = f"<tool_response>\n{snippets_for_all_queries}\n</tool_response>"
        except Exception as e:
            log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to execute search for tool {tool_name}")
            search_result = f"<tool_response>\nFailed to execute search. Please check if the search query is valid.\n</tool_response>"
            search_elapsed_time = 0.0
            time_stats["search"].append(search_elapsed_time)
        finally:
            messages.append({
                "role": "tool",
                "content": search_result,
                "time": {
                    "name": "search",
                    "elapsed_time": search_elapsed_time
                }
            })
        
        return False  # no need to break
    
    async def _execute_visit_tool(self, tool_args: dict, tool_name: str, idx: int, iteration: int,
                                    messages: list, msg_len_before_assistant: int, time_stats: dict) -> bool:
        """
        执行访问工具（通用方法，支持 visit 和 browse_page）
        
        Returns:
            bool: True means need to break (parameter parsing failed), False means continue
        """
        # Stage 2a: Parse visit parameters
        try:
            urls = tool_args['url']
            goal = tool_args['goal']

            if isinstance(urls, str) and urls.strip().startswith("[") and urls.strip().endswith("]"):
                urls = json.loads(urls.strip())
            # case2: model gives urls as str, unify to list[str]
            elif isinstance(urls, str):
                urls = [urls]
            # case3: model gives urls as list[str], use directly
            else:
                pass
        except Exception as e:
            log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to parse tool call arguments for tool {tool_name}")
            logger.warning(f"Query {idx}, Iteration {iteration}: Failed to parse [{tool_name}] tool call arguments. Skip this iteration.")
            messages[:] = messages[:msg_len_before_assistant]
            return True  # need to break
        
        # Stage 2b: Execute web page visit
        try:
            visit_start_time = time.time()
            
            # Concurrently access URLs
            semaphore = asyncio.Semaphore(10)
            async def fetch_content(url: str, idx: int) -> WebContent:
                try:
                    async with semaphore:
                        return await self.web_crawl_tool.crawl(url)
                except Exception as exc:
                    logger.error("Failed to fetch content for %s: %s", url, exc)
                    logger.error(traceback.format_exc())
                    return WebContent(text_content="[Visit] Failed to read page.")
            
            web_contents = await asyncio.gather(
                *(fetch_content(url, idx) for idx, url in enumerate(urls))
            )
            
            # Use backup crawl tool
            if self.web_crawl_tool_backup is not None:
                for idx_web, (url, web_content) in enumerate(zip(urls, web_contents)):
                    if web_content.text_content is None \
                        or web_content.text_content.startswith("[Visit] Failed to read page."):
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
            
            # Call LLM to summarize web page content
            summarize_start_time = time.time()
            summarized_contents = await asyncio.gather(
                *(self.web_summarize_tool.summarize(
                    url=urls[idx],
                    raw_content=web_content.text_content,
                    goal=goal
                ) for idx, web_content in enumerate(web_contents))
            )
            summarize_elapsed_time = time.time() - summarize_start_time
            time_stats["summarize"].append(summarize_elapsed_time)
            
            visit_elapsed_time = time.time() - visit_start_time
            time_stats["visit"].append(visit_elapsed_time)
            
            result = "\n=======\n".join(summarized_contents).strip()
            visit_result = f"<tool_response>\n{result}\n</tool_response>"
        except Exception as e:
            log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to execute {tool_name} for tool {tool_name}")
            visit_result = f"<tool_response>\nWeb visit failed. The access to the URL is blocked or denied.\n</tool_response>"
            visit_elapsed_time = 0.0
            summarize_elapsed_time = 0.0
            time_stats["visit"].append(visit_elapsed_time)
            time_stats["summarize"].append(summarize_elapsed_time)
        finally:
            messages.append({
                "role": "tool",
                "content": visit_result,
                "time": [
                    {"name": "visit", "elapsed_time": visit_elapsed_time},
                    {"name": "summarize", "elapsed_time": summarize_elapsed_time}
                ]
            })
        
        return False  # no need to break

    async def process_query(self, idx: int, user_query: str, ground_truth: str = None, 
                            dataset_id: str = None, sample_idx: int = 0, override: bool = False):
        # Generate different save paths based on sample_idx
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
            # Check if saved results or checkpoint exists
            start_iteration = 0
            resume_from_checkpoint = False
            start_time = time.time()
            
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    saved_results = json.load(f)
                
                # Check if checkpoint needs to be resumed
                if saved_results.get("resume", False):
                    # Need to resume from checkpoint
                    logger.info(f"Problem {idx} sample {sample_idx} found checkpoint at iteration {saved_results.get('current_iteration', 0)}. Resuming...")
                    resume_from_checkpoint = True
                    # Restore messages
                    messages = saved_results["messages"]
                    # Restore iteration (start from next iteration)
                    start_iteration = saved_results.get("current_iteration", 0) + 1
                    # Restore elapsed time (start from accumulated time in checkpoint)
                    start_time = time.time() - saved_results.get("processing_time_seconds", 0)

                    # Restore time_stats
                    time_stats = {
                        "assistant": saved_results["time_summary"]["assistant"]["details"].copy(),
                        "search": saved_results["time_summary"]["search"]["details"].copy(),
                        "visit": saved_results["time_summary"]["visit"]["details"].copy(),
                        "summarize": saved_results["time_summary"]["summarize"]["details"].copy(),
                        "python": saved_results["time_summary"]["python"]["details"].copy(),
                        "google_scholar": saved_results["time_summary"]["google_scholar"]["details"].copy(),
                        "google_maps": saved_results["time_summary"]["google_maps"]["details"].copy()
                    }
                    
                    logger.info(f"Problem {idx} sample {sample_idx} resumed from iteration {start_iteration}")
                else:
                    # Completed results
                    if override:
                        logger.info(f"Problem {idx} sample {sample_idx} already solved. Overriding...")
                    else:
                        logger.info(f"Problem {idx} sample {sample_idx} already solved. Skipping...")
                        return saved_results

            # If not resuming from checkpoint, initialize new message list
            if not resume_from_checkpoint:
                logger.info(f"Processing query {idx} started")
                current_date = datetime.now().strftime("%Y-%m-%d")
                system_prompt = self.model_args["system_prompt"]
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]

                # Initialize time statistics
                time_stats = {
                    "assistant": [], "search": [], "visit": [], "summarize": [], "python": [], "google_scholar": [], "google_maps": []
                }
            
            # Start the real Agent Loop
            force_answer = False
            termination = None
            failed_iterations = []
            for iteration in range(start_iteration, self.config["agent"]["max_iter"]):
                logger.info("Query %s, Iteration %s started, max iteration is %s", idx, iteration, self.config["agent"]["max_iter"])
                # Assistant generation step
                try:
                    start_time = time.time()
                    content = await async_chat_completion(self._filter_time_from_messages(messages), self.config["llm"]["local"]["model_name"], self.llm_endpoint)
                    assistant_elapsed_time = time.time() - start_time
                    time_stats["assistant"].append(assistant_elapsed_time)
                except Exception as e:
                    failed_iterations.append("LLM Call Failed")
                    log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: LLM call failed - . Skipping this iteration.")
                    continue

                # Parse tool calls
                # Model may hallucinate after tool calls, generating observations by itself
                if '<tool_response>' in content:
                    pos = content.find('<tool_response>')
                    content = content[:pos]

                # Case1: Parse tool calls
                if '<tool_call>' in content and '</tool_call>' in content:
                    assistant_msg = {
                        "role": "assistant", 
                        "content": content,
                        "time": {
                            "name": "assistant",
                            "elapsed_time": assistant_elapsed_time
                        }
                    }
                    msg_len_before_assistant = len(messages)
                    messages.append(assistant_msg)

                    tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                    try:
                        # RedSearcher Agent style tool calls
                        if "PythonInterpreter".lower() in tool_call.lower():
                            # Stage 1: Parse Python code
                            try:
                                code_raw = content.split('<tool_call>')[1].split('</tool_call>')[0].split('<code>')[1].split('</code>')[0].strip()
                                code = self._post_process_code(code_raw)
                            except Exception as e:
                                log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to parse tool call arguments for PythonInterpreter")
                                logger.warning(f"Query {idx}, Iteration {iteration}: Failed to parse [PythonInterpreter] tool call arguments. Skip this iteration.")
                                messages = messages[:msg_len_before_assistant]
                                continue
                            
                            # Stage 2: Execute Python code
                            try:
                                python_start_time = time.time()
                                result = await run_python(code, **self.config["tools"]["python"])
                                python_elapsed_time = time.time() - python_start_time
                                time_stats["python"].append(python_elapsed_time)
                                code_execution_result = f"<tool_response>\n{result.strip()}\n</tool_response>"
                            except Exception as e:
                                log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to execute Python Code.")
                                code_execution_result = f"<tool_response>\nFailed to execute python\n</tool_response>"
                                python_elapsed_time = 0.0
                                time_stats["python"].append(python_elapsed_time)
                            finally:
                                messages.append({
                                    "role": "tool",
                                    "content": code_execution_result,
                                    "time": {
                                        "name": "python",
                                        "elapsed_time": python_elapsed_time
                                    }
                                })
                        else:
                            # RedSearcher Agent style other tool calls
                            # Stage 1: Parse tool call JSON
                            try:
                                tool_call_parsed = json5.loads(tool_call)
                                tool_name = tool_call_parsed.get('name', '')
                                tool_args = tool_call_parsed.get('arguments', {})
                            except Exception as e:
                                log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to parse tool call JSON")
                                logger.warning(f"Query {idx}, Iteration {iteration}: Failed to parse tool call JSON. Skip this iteration.")
                                messages = messages[:msg_len_before_assistant]
                                continue
                            
                            if tool_name == 'search':
                                should_break = await self._execute_search_tool(
                                    tool_args, tool_name, idx, iteration,
                                    messages, msg_len_before_assistant, time_stats
                                )
                                if should_break:
                                    continue
                            
                            elif tool_name == 'visit':
                                should_break = await self._execute_visit_tool(
                                    tool_args, tool_name, idx, iteration,
                                    messages, msg_len_before_assistant, time_stats
                                )
                                if should_break:
                                    continue
                            elif tool_name == 'google_scholar':
                                should_break = await self._execute_google_scholar_tool(
                                    tool_args, tool_name, idx, iteration,
                                    messages, msg_len_before_assistant, time_stats
                                )
                                if should_break:
                                    continue
                            elif tool_name == 'google_maps':
                                should_break = await self._execute_google_maps_tool(
                                    tool_args, tool_name, idx, iteration,
                                    messages, msg_len_before_assistant, time_stats
                                )
                                if should_break:
                                    continue
                            else:
                                logger.error(f"Tool {tool_name} is not implemented")
                                logger.warning(f"Query {idx}, Iteration {iteration}: Tool {tool_name} is not implemented. Skip this iteration.")
                                messages = messages[:msg_len_before_assistant]
                                continue

                    except Exception as e:
                        log_exception_details(logger, e, f"Query {idx}, Iteration {iteration}: Failed to parse tool call string.")
                        logger.warning(f"Query {idx}, Iteration {iteration}: Failed to parse tool call JSON. Skip this iteration.")
                        messages = messages[:msg_len_before_assistant]
                        continue

                
                # Case2: Model found answer or generation error occurred
                else:
                    if "<answer>" in content:
                        assistant_msg = {
                            "role": "assistant",
                            "content": content,
                            "time": {
                                "name": "assistant",
                                "elapsed_time": assistant_elapsed_time
                            }
                        }
                        termination = "answer"
                        messages.append(assistant_msg)
                        break
                    else:
                        logger.warning(f"Query {idx}, Iteration {iteration}: Model generated content but no answer found. Skipping this iteration.\n{content[:150]}")
                        continue
            
                # Consider context and truncation issues
                if iteration == self.config["agent"]["max_iter"] - 1:
                    termination = "max_iter"
                    force_answer = True
                    logger.info(f"Query {idx}: The number of iterations reached the maximum limit: {self.config['agent']['max_iter']}. Force answering the question.")
                
                token_count = self._count_tokens(messages)
                max_tokens = 120 * 1000
                if token_count > max_tokens:
                    termination = "context length"
                    force_answer = True
                    logger.info(f"Query {idx}: The context length reached the maximum limit: {token_count} tokens. Rollback messages to fit and force answering the question.")

                if force_answer:
                    # Backtrack to find messages with length less than max and ending with assistant or tool role
                    truncated_messages = []
                    for msg in messages:
                        # Try to add current message
                        truncated_messages.append(msg)
                        # Calculate token count for entire message list
                        current_token_count = self._count_tokens(truncated_messages)
                        if current_token_count > max_tokens:
                            # If exceeds limit, remove just added message and stop
                            truncated_messages.pop()
                            break

                    # Search from end to find truncation point ending with assistant or tool role
                    for i in range(len(truncated_messages)-1, -1, -1):
                        if truncated_messages[i]["role"] in {"assistant"}:
                            truncated_messages = truncated_messages[:i+1]
                            break
                
                    messages = truncated_messages
                    messages.append({
                        "role": "user", 
                        "content": "The context length reached the maximum context length / iteration limit. Please answer the question based on previous context. The answer should be wrapped in <answer> </answer> tags."
                    })

                    force_answer_start_time = time.time()
                    content = await async_chat_completion(self._filter_time_from_messages(messages), self.config["llm"]["local"]["model_name"], self.llm_endpoint)
                    force_answer_time_elapsed = time.time() - force_answer_start_time
                    time_stats["assistant"].append(force_answer_time_elapsed)
                    assistant_msg = {
                        "role": "assistant",
                        "content": content,
                        "time": {
                            "name": "assistant",
                            "elapsed_time": force_answer_time_elapsed
                        }
                    }
                    messages.append(assistant_msg)
                    break

                # Save running results periodically for timeout recovery
                checkpoint_interval = 10 
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
           
            # Agent Loop ends
            if "<answer>" in messages[-1]["content"] and "</answer>" in messages[-1]["content"]:
                prediction = messages[-1]["content"].split("<answer>")[1].split("</answer>")[0]
            else:
                prediction = messages[-1]["content"]
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
            # Count tokens consumption
            try:
                tokens_count = self._count_tokens(messages)
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

