"""
XBench Evaluation Implementation
"""

import re
from typing import Optional, Dict
from src.eval.base_judge import BaseJudge, local_judge, remote_gemini_judge


# LLM Judge prompt (Chinese) - from eval_grader.py
LLM_JUDGE_PROMPT = """
你是一个通用人工智能助手。根据下面给出的[正确答案], 判断以下对[原问题]的[回答]的回答是否正确。

[原问题]: {question}

[正确答案]: {correct_answer}

[回答]:{response}

你的判断必须按照以下格式和标准进行:

最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案, 则填写'无'。

解释: 根据[原问题]解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与[正确答案]之间是否存在实质性差异, 不要评论题目的背景, 不要尝试重新解题, 不要为任何不同于[正确答案]的答案辩护, 只专注于判断答案是否一致。

结论: 如果[最终答案]与上方给出的[正确答案]一致, 或者在数值题目中处于可接受的微小误差范围内, 则填写'正确'; 否则（即存在任何不一致、歧义、不等价或提取出的答案错误的情况）填写'错误'。
""".strip()


def parse_xbench_judge_output(judge_response: str) -> dict:
    """
    Parse XBench Judge output content
    
    Returns:
        dict: Evaluation result, aligned with BaseJudge format
    """
    if not judge_response or judge_response == "PARSE FAILED":
        return "PARSE FAILED"

    # Extract grader conclusion
    extract_match = re.search(r'最终答案:*(.*)', judge_response)
    extracted_answer = extract_match.group(1).strip() if extract_match else ""
    
    correct_match = re.search(r"结论:*\s*(正确|错误)", judge_response)
    correct_result = correct_match.group(1).strip() if correct_match else "错误"
    correct = "yes" if (correct_result == "正确") else "no"
    
    explain_match = re.search(r"解释:*(.*?)(?:结论:|$)", judge_response, re.DOTALL)
    explanation = explain_match.group(1).strip() if explain_match else ""
    
    score = 1.0 if (correct_result == "正确") else 0.0

    return {
        "extracted_final_answer": extracted_answer,
        "reasoning": explanation,
        "correct": correct,
        "confidence": None,
        "accuracy": score
    }
    

class XBenchJudge(BaseJudge):
    """
    XBench Judge
    Uses LLM judge to determine if answer is correct (Chinese evaluation)
    Implementation logic based on eval_grader.py
    """
    def __init__(self, remote: bool = False, model_name: str = None, endpoint: str = None):
        super().__init__(remote, model_name, endpoint)
        self.judge_template = LLM_JUDGE_PROMPT
    
    def judge(self, question: str, response: str, correct_answer: str) -> Dict:
        """
        Evaluate XBench task
        
        Args:
            question: Question text
            correct_answer: Ground truth answer
            response: Model response
        
        Returns:
            Dict: Evaluation result details
        """
        # First try simple matching
        simple_match = re.search(r'最终答案:*(.*)', response)
        simple_match_result = self._parse_match_result(simple_match)
        
        if simple_match_result and simple_match_result == correct_answer:
            # Direct match successful, no LLM judge needed
            return {
                "extracted_final_answer": simple_match_result,
                "reasoning": "Answer completely correct, no LLM Judge needed",
                "correct": "yes",
                "confidence": None,
                "accuracy": 1.0
            }
        
        # Use LLM Judge
        judge_prompt = self.judge_template.format(question=question, response=response, correct_answer=correct_answer)
        if self.remote:
            result = remote_gemini_judge(judge_prompt, parse_xbench_judge_output)
        else:
            result = local_judge(judge_prompt, self.endpoint, self.model_name, parse_xbench_judge_output)
        
        return result
    
    def _parse_match_result(self, match) -> str:
        """
        Parse Match result
        
        Args:
            match: Regex match result
            
        Returns:
            str: Extracted content
        """
        if match is None:
            return ""
        
        match_str = match.group(0)
        
        try:
            target = match_str.split(':')[1].strip()
            return target
        except Exception:
            return match_str  # Return original result
        