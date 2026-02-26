"""
DirectQAVerifier: Direct answer questions without search.
"""

import os                       
import sys
import json
import time
import traceback
from transformers import AutoTokenizer

if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-3])
    sys.path.append(base_path)

from src.agents.base import AgentBase
from src.eval import get_judge
from utils.logger import get_logger
from src.llms_async import async_chat_completion

logger = get_logger(__name__)

def log_exception_details(logger, exception, context=""):
    """Record detailed exception information"""
    logger.error(f"{context}Exception occurred:")
    logger.error(f"  Type: {type(exception).__name__}")
    logger.error(f"  Message: {str(exception)}")
    logger.error(f"  Traceback:\n{traceback.format_exc()}")


# Directly verify QA correctness without retrieval
class DirectQAVerifier(AgentBase):
    def __init__(self, config_path, **kwargs):
        super().__init__(config_path, **kwargs)
        self._init_tokenizer()

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2")

    def _count_tokens(self, messages):
        encoding = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
        return len(encoding)

    async def process_query(self, idx: int, user_query: str, ground_truth: str = None, 
                            dataset_id: str = None, sample_idx: int = 0, override: bool = False):
        """
        Simple single-turn QA verification without tool calls, only requires one round of model response.
        """
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

        try:
            start_time = time.time()
            
            # Check if already completed
            if os.path.exists(save_path):
                with open(save_path, "r") as f:
                    saved_results = json.load(f)
                
                # Check if override is needed
                if not saved_results.get("resume", False):
                    if override:
                        logger.info(f"Problem {idx} sample {sample_idx} already solved. Overriding...")
                    else:
                        logger.info(f"Problem {idx} sample {sample_idx} already solved. Skipping...")
                        return saved_results
            
            logger.info(f"Processing query {idx} sample {sample_idx} started (DirectQAVerifier)")
            
            # Build simple system prompt (no tools needed)
            system_prompt = """You are a helpful AI assistant. Please answer the user's question accurately and concisely.

When you have your final answer ready, enclose it within <answer></answer> tags."""
            
            # Initialize message list
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
            
            # Call LLM once
            logger.info(f"Query {idx} sample {sample_idx}: Calling LLM for single-turn response")
            llm_start_time = time.time()
            try:
                content = await async_chat_completion(
                    messages, 
                    self.config["llm"]["local"]["model_name"], 
                    self.llm_endpoint
                )
                llm_elapsed_time = time.time() - llm_start_time
            except Exception as e:
                log_exception_details(logger, e, f"Query {idx} sample {sample_idx}: LLM call failed")
                raise e
            
            # Add assistant response to messages
            messages.append({
                "role": "assistant",
                "content": content,
                "time": {
                    "name": "assistant",
                    "elapsed_time": llm_elapsed_time
                }
            })
            
            # Extract prediction
            if "<answer>" in content and "</answer>" in content:
                prediction = content.split("<answer>")[1].split("</answer>")[0].strip()
            else:
                prediction = content.strip()
            
            logger.info(f"Query {idx} sample {sample_idx}: Got prediction, calling LLM-as-judge")
            
            # Get judge and evaluate
            remote = self.config.get("judge", {}).get("remote", True)
            model_name = self.config.get("judge", {}).get("local", {}).get("model_name")
            endpoint = self.config.get("judge", {}).get("local", {}).get("endpoint")
            judger = get_judge(dataset_id, remote, model_name, endpoint)
            judge_result = judger.judge(user_query, prediction, ground_truth)
            
            # Count tokens
            try:
                tokens_count = self._count_tokens(messages)
            except Exception as e:
                logger.warning(f"Query {idx} sample {sample_idx}: Failed to count tokens: {e}")
                tokens_count = None
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Build time summary
            time_summary = {
                "assistant": {
                    "count": 1,
                    "total_time": llm_elapsed_time,
                    "details": [llm_elapsed_time]
                },
                "search": {
                    "count": 0,
                    "total_time": 0,
                    "details": []
                },
                "visit": {
                    "count": 0,
                    "total_time": 0,
                    "details": []
                },
                "summarize": {
                    "count": 0,
                    "total_time": 0,
                    "details": []
                },
                "python": {
                    "count": 0,
                    "total_time": 0,
                    "details": []
                },
                "total_time": total_time
            }
            
            # Build result
            result = {
                "idx": idx,
                "dataset_id": dataset_id,
                "question": user_query,
                "answer": ground_truth,
                "messages": messages,
                "prediction": prediction,
                "termination": "single_turn_completion",
                "processing_time_seconds": total_time,
                "time_summary": time_summary,
                "llm_as_judge": judge_result,
                "turns": 1,
                "tokens_count": tokens_count
            }
            
            # Save result
            with open(save_path, "w") as f:
                json.dump(result, f, indent=4)
            
            logger.info(f"Query {idx} sample {sample_idx}: Completed. Accuracy: {judge_result.get('accuracy', 'N/A')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Query {idx} sample {sample_idx}: Failed to process query: {e}")
            log_exception_details(logger, e, f"Query {idx} sample {sample_idx}: Failed to process query")
            raise e

