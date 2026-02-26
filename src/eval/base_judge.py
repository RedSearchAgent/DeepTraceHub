"""
Base Judge Interface Class
Provides unified evaluation interface, all specific evaluation implementations should inherit this class
"""
import os
import re
import sys
import logging
from typing import Dict

if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2])
    sys.path.append(base_path)

from abc import ABC
from src.llms import chat_completion_gemini
from src.llms_async import async_chat_completion, sync_chat_completion
from prompt.system_prompt import LLM_AS_JUDGE_BROWSECOMP_OFFICAL_PROMPT, LLM_AS_JUDGE_BROWSECOMP_TONGYI_EN_PROMPT

logger = logging.getLogger(__name__)
_llm_as_judge_re_pattern = r"^(extracted_final_answer|reasoning|correct|confidence):\s*(.+)$"


def parse_judge_output(output: str) -> Dict:
    try:
        json_output = dict(re.findall(_llm_as_judge_re_pattern, output, re.MULTILINE))
        json_output = {k: v.strip() for k, v in json_output.items()}
        json_output["accuracy"] = float(json_output["correct"] == "yes")
        return json_output
    except Exception as e:
        return "PARSE FAILED"

def local_judge_low_fn(
    prompt: str, endpoint: str, model_name: str, **kwargs
) -> Dict:
    messages = [{
        "role": "user", "content": prompt
    }]
    max_attempts = 5
    for attempt in range(max_attempts):
        output = sync_chat_completion(
            messages=messages,
            model=model_name,
            endpoint=endpoint
        )
        output = output.strip()
        if output.startswith("A"):
            return {
                "extracted_final_answer": None,
                "reasoning": None,
                "correct": "yes",
                "confidence": None,
                "accuracy": 1.0
            }
        elif output.startswith("B"):
            return {
                "extracted_final_answer": None,
                "reasoning": None,
                "correct": "no",
                "confidence": None,
                "accuracy": 0.0
            }
        else:
            logger.warning(f"parse_judge_output failed on attempt {attempt+1}/{max_attempts}, retrying...")
            continue
    logger.error("Failed to parse judge output.")
    return {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": "no",
        "confidence": None,
        "accuracy": 0.0
    }

def local_judge(prompt: str, endpoint: str, model_name: str, parsed_func=None) -> Dict:
    messages = [{
        "role": "user", "content": prompt
    }]
    max_attempts = 5
    for attempt in range(max_attempts):
        output = sync_chat_completion(
            messages=messages,
            model=model_name,
            endpoint=endpoint
        )
        if parsed_func is not None:
            result = parsed_func(output)
        else:
            result = parse_judge_output(output)
        if result == "PARSE FAILED":
            logger.warning(f"parse_judge_output failed on attempt {attempt+1}/{max_attempts}, retrying...")
            continue
        else:
            return result
    logger.error("Failed to parse judge output.")
    return {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": "no",
        "confidence": None,
        "accuracy": 0.0
    }


def remote_gemini_judge(prompt: str, parsed_func=None) -> Dict:
    max_attempts = 3
    for attempt in range(max_attempts):
        output = chat_completion_gemini(prompt)
        # Note: parsed_func return format needs to be consistent with parse_judge_output
        if parsed_func is not None:
            result = parsed_func(output)
        else:
            result = parse_judge_output(output)
        if result == "PARSE FAILED":
            logger.warning(f"parse_judge_output failed on attempt {attempt+1}/{max_attempts}, retrying...")
            continue
        else:
            return result
    return {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": "no",
        "confidence": None,
        "accuracy": 0.0
    }


async def async_local_judge(question: str, response: str, correct_answer: str, endpoint: str, model_name: str) -> Dict:
    input_prompt = LLM_AS_JUDGE_BROWSECOMP_OFFICAL_PROMPT.format(question=question, response=response, correct_answer=correct_answer)
    messages = [{
        "role": "user", "content": input_prompt
    }]
    max_attempts = 5
    for attempt in range(max_attempts):
        output = await async_chat_completion(messages, model_name, endpoint, n=1)
        result = parse_judge_output(output)
        if result == "PARSE FAILED":
            logger.warning(f"parse_judge_output failed on attempt {attempt+1}/{max_attempts}, retrying...")
            continue
        else:
            return result
    return {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": "no",
        "confidence": None,
        "accuracy": 0.0
    }


class BaseJudge(ABC):
    """
    Base judge class defining unified evaluation interface
    Applicable for: BrowseComp/BrowseComp-zh/HLE or similar tasks
    """
    
    def __init__(self, remote: bool = False, model_name: str = None, endpoint: str = None):
        """
        Initialize judge
        
        Args:
            remote: Whether to use remote evaluation
            model_name: Model name
            endpoint: Evaluation endpoint
        """
        self.remote = remote
        self.model_name = model_name
        self.endpoint = endpoint
        self.judge_template = LLM_AS_JUDGE_BROWSECOMP_TONGYI_EN_PROMPT
    
    def judge(self, question: str, response: str, correct_answer: str) -> Dict:
        judge_prompt = self.judge_template.format(question=question, response=response, correct_answer=correct_answer)

        if self.remote:
            return remote_gemini_judge(judge_prompt)
        else:
            return local_judge_low_fn(judge_prompt, self.endpoint, self.model_name)
