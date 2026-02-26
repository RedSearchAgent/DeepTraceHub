"""
GAIA Evaluation Implementation
"""

import re
import string
import warnings
from typing import Optional, Dict, Any
from src.eval.base_judge import BaseJudge, local_judge, remote_gemini_judge, LLM_AS_JUDGE_BROWSECOMP_OFFICAL_PROMPT


class GAIAJudge(BaseJudge):
    """
    GAIA Judge
    Uses rule-based string matching and number comparison
    """
    
    def judge(self, question: str, response: str, correct_answer: str) -> Dict:
        """
        Evaluate GAIA task
        
        Args:
            question: Question text
            correct_answer: Ground truth answer
            response: Model response (should be content extracted from <answer> tag)
        
        Returns:
            Dict: Evaluation result details
        """
        try:
            # Extract boxed content from answer_text
            extracted = self.extract_from_answer_xml(response)
            
            # If extraction fails, use original answer
            if not extracted:
                extracted = response
                extraction_method = "original_answer"
                # TODO: Use an LLM-as-judge based evaluation method
                judge_prompt = LLM_AS_JUDGE_BROWSECOMP_OFFICAL_PROMPT.format(question=question, response=response, correct_answer=correct_answer)

                if self.remote:
                    result = remote_gemini_judge(judge_prompt)
                else:
                    result = local_judge(judge_prompt, self.endpoint, self.model_name)
                
                result["extraction_method"] = "LLM-as-Judge"
                return result
            else:
                extraction_method = "extracted_from_<answer>"
                is_correct = self._question_scorer(extracted, correct_answer)

                acc = 1.0 if is_correct else 0.0
                correct = "yes" if is_correct else "no"

                return {
                    "extracted_final_answer": extracted,
                    "reasoning": None,
                    "correct": correct,
                    "confidence": None,
                    "accuracy": acc,
                    "extraction_method": extraction_method
                }
        except Exception as e:
            print(f" [ERROR] GAIA evaluation failed: {e}")
            return {
                "extracted_final_answer": response,
                "reasoning": f"GAIA evaluation exception: {str(e)}",
                "correct": "no",
                "confidence": None,
                "accuracy": 0.0,
                "error": str(e)
            }
    
    def extract_from_answer_xml(self, text):
        """
        Extract answer from <answer> </answer>
        
        Args:
            text: Input text
            
        Returns:
            str: Extracted answer
        """
        pattern = r"<answer>(.*?)</answer>"
        match = re.search(pattern, text)
        if match:
            return match.group(1)
        return ""
    
    def _normalize_number_str(self, number_str: str) -> float:
        """
        Normalize number string
        
        Args:
            number_str: Number string
            
        Returns:
            float: Normalized number
        """
        # Remove common units and commas
        for char in ["$", "%", ","]:
            number_str = number_str.replace(char, "")
        try:
            return float(number_str)
        except ValueError:
            print(f" [WARNING] String {number_str} cannot be normalized to number")
            return float("inf")
    
    def _split_string(self, s: str, char_list: list = [",", ";"]) -> list:
        """
        Split string by specified characters
        
        Args:
            s: Input string
            char_list: List of delimiters
            
        Returns:
            list: List of split strings
        """
        pattern = f"[{''.join(char_list)}]"
        return re.split(pattern, s)
    
    def _normalize_str(self, input_str: str, remove_punct: bool = True) -> str:
        """
        Normalize string: remove spaces, punctuation, convert to lowercase
        
        Args:
            input_str: Input string
            remove_punct: Whether to remove punctuation
            
        Returns:
            str: Normalized string
        """
        # Remove all spaces
        no_spaces = re.sub(r"\s", "", input_str)
        
        # Remove punctuation (if specified)
        if remove_punct:
            translator = str.maketrans("", "", string.punctuation)
            return no_spaces.lower().translate(translator)
        else:
            return no_spaces.lower()
    
    def _is_float(self, element: Any) -> bool:
        """
        Check if element can be converted to float
        
        Args:
            element: Element to check
            
        Returns:
            bool: Whether it can be converted to float
        """
        try:
            float(element)
            return True
        except (ValueError, TypeError):
            return False
    
    def _question_scorer(self, model_answer: str, ground_truth: str) -> bool:
        """
        Scoring function, compare model answer with ground truth
        
        Args:
            model_answer: Model answer
            ground_truth: Ground truth answer
            
        Returns:
            bool: Whether correct
        """
        if model_answer is None:
            model_answer = "None"
        
        # If ground truth is a number
        if self._is_float(ground_truth):
            print(f" [INFO] Evaluating {model_answer} as number")
            normalized_answer = self._normalize_number_str(model_answer)
            return normalized_answer == float(ground_truth)
        
        # If ground truth is a list (contains comma or semicolon)
        elif any(char in ground_truth for char in [",", ";"]):
            print(f" [INFO] Evaluating {model_answer} as comma-separated list")
            
            gt_elems = self._split_string(ground_truth)
            ma_elems = self._split_string(model_answer)
            
            # Check if lengths are the same
            if len(gt_elems) != len(ma_elems):
                warnings.warn(
                    "Answer list lengths differ, returning False", UserWarning
                )
                return False
            
            # Compare elements one by one
            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if self._is_float(gt_elem):
                    normalized_ma_elem = self._normalize_number_str(ma_elem)
                    comparisons.append(normalized_ma_elem == float(gt_elem))
                else:
                    # Don't remove punctuation as comparison may include punctuation
                    comparisons.append(
                        self._normalize_str(ma_elem, remove_punct=False)
                        == self._normalize_str(gt_elem, remove_punct=False)
                    )
            return all(comparisons)
        
        # If ground truth is a string
        else:
            print(f" [INFO] Evaluating {model_answer} as string")
            return self._normalize_str(model_answer) == self._normalize_str(ground_truth)

