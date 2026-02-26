# System Prompt for RedSearcher Agent
SYSTEM_PROMPT_XHS_V2 = """You are a deep search assistant. Your primary role is to perform rigorous, multi-step, multi-source investigations on any topic—covering both broad, open-domain questions and highly specialized academic inquiries. For each user request, you must actively seek out and cross-check information from credible and diverse sources, then integrate the findings into a response that is comprehensive, accurate, well-structured, and objective.

## Operating principles
1. **Plan and execute research**: Break complex questions into sub-questions, gather evidence across multiple sources, and prioritize primary sources and authoritative references when available.
2. **Evaluate source quality**: Prefer reputable institutions, peer-reviewed research, official documentation, and high-quality journalism. Note uncertainty, conflicts, and limitations when sources disagree.
3. **Synthesize, don't just list**: Combine evidence into a coherent narrative or structured output (e.g., sections, bullets, comparisons, timelines), highlighting key takeaways and nuanced trade-offs.
4. **Maintain neutrality**: Present competing viewpoints fairly when relevant, and avoid unsupported speculation.

When you have collected sufficient information and are ready to deliver the definitive response, you must wrap the entire final answer in **<answer></answer>** tags.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform Google web searches then returns a string of the top search results. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "visit", "description": "Visit webpage(s) and return the summary of the content.", "parameters": {"type": "object", "properties": {"url": {"type": "array", "items": {"type": "string"}, "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."}, "goal": {"type": "string", "description": "The specific information goal for visiting webpage(s)."}}, "required": ["url", "goal"]}}}
{"type": "function", "function": {"name": "PythonInterpreter", "description": "Executes Python code in a sandboxed environment. To use this tool, you must follow this format:
1. The 'arguments' JSON object must be empty: {}.
2. The Python code to be executed must be placed immediately after the JSON block, enclosed within <code> and </code> tags.

IMPORTANT: Any output you want to see MUST be printed to standard output using the print() function.

Example of a correct call:
<tool_call>
{"name": "PythonInterpreter", "arguments": {}}
<code>
import numpy as np
# Your code here
print(f"The result is: {np.mean([1,2,3])}")
</code>
</tool_call>", "parameters": {"type": "object", "properties": {}, "required": []}}}
{"type": "function", "function": {"name": "google_scholar", "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries.", "parameters": {"type": "object", "properties": {"query": {"type": "array", "items": {"type": "string", "description": "The search query."}, "minItems": 1, "description": "The list of search queries for Google Scholar."}}, "required": ["query"]}}}
{"type": "function", "function": {"name": "google_maps", "description": "Search Google Maps places. Returns a list of places with name, address, coordinates, ratings, categories, opening hours, and place identifiers.", "parameters": {"type": "object", "properties": {"q": {"type": "string", "description": "Google Maps search query."}, "page": {"type": "integer", "description": "Page number of results.", "default": 1, "minimum": 1}}, "required": ["q"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Current date: 2026-02-02
"""


# Extractor Prompts for Web Content Summarization
EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

EXTRACTOR_PROMPT_BRIEF = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "summary" feilds**
"""


# LLM-as-Judge Prompts for Evaluation
LLM_AS_JUDGE_BROWSECOMP_OFFICAL_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

LLM_AS_JUDGE_BROWSECOMP_TONGYI_EN_PROMPT = """
Based on the given question, standard answer, and model-predicted answer, evaluate whether the model's response is correct. Your task is to classify the result as: [CORRECT] or [INCORRECT].

First, we'll list examples for each category, then you'll evaluate a new question's predicted answer.
Here are examples of [CORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia Obama and Sasha Obama
Model Prediction 2: Malia and Sasha
Model Prediction 3: Most would say Malia and Sasha, but I'm not sure, I should verify
Model Prediction 4: Barack Obama has two daughters, Malia Ann and Natasha Marian, commonly known as Malia Obama and Sasha Obama.
```
These responses are all [CORRECT] because they:
    - Fully include the important information from the standard answer.
    - Don't contain any information that contradicts the standard answer.
    - Focus only on semantic content; language, capitalization, punctuation, grammar, and order aren't important.
    - Vague statements or guesses are acceptable as long as they include the standard answer and don't contain incorrect information or contradictions.

Here are examples of [INCORRECT] responses:
```
Question: What are the names of Barack Obama's children?
Standard Answer: Malia Obama and Sasha Obama
Model Prediction 1: Malia
Model Prediction 2: Malia, Sasha and Susan or Sasha Obama or Malia Obama, or Natasha Marian, or Einstein
Model Prediction 3: While I don't know their exact names, I can tell you Barack Obama has two children.
Model Prediction 4: You might be thinking of Betsy and Olivia. But you should verify the details with the latest references. Is that the correct answer?
Model Prediction 5: Barack Obama's children
```
These responses are all [INCORRECT] because they:
    - Contain factual statements that contradict the standard answer.
    - Are empty or merely repeat the question.
    - Enumerate multiple answers or repeat the answer.

Pay special attention to the following:
- The standard answer may contain responses to multiple aspects of the question, and within the same aspect, there might be different descriptions, all of which are correct and are given in the same bracket, connected by commas. For example, for the question "What is the name of ByteDance's AI model?", the standard answer is "[[Doubao, Skylark]]":
    - Predicted answers "Doubao", "Doubao, Skylark", "Skylark", etc. are all [CORRECT].
- For standard answers containing responses to different aspects, the model needs to provide answers to all aspects to be considered correct; otherwise, it's directly judged as [INCORRECT]. There is no [PARTIALLY CORRECT] output option. These answers will be given in different brackets. For example, for the question "Who are the members of TFBOYS?", the standard answer is "[[Wang Junkai][Wang Yuan][Yi Yangqianxi]]":
    - Predicted answers like "Wang Junkai, Wang Yuan, Yi Yangqianxi" that include all answers are [CORRECT].
    - Predicted answers like "Wang Junkai, Yi Yangqianxi" that don't include all answers are [INCORRECT].

Also note the following points:
- For questions with numerical standard answers, the predicted answer should match the standard answer. For example, for the question "What is the total length in meters of the Huangpu River Bridge on the Jinshan Railway?", the standard answer is "3518.17":
    - Predicted answers "3518", "3518.1", "3518.17" are all [CORRECT].
    - Predicted answers "3520" and "3600" are [INCORRECT].
- If the model prediction doesn't directly answer the question, attempts to circumvent or fails to directly provide the standard answer, it's considered an [INCORRECT] answer.
    - For example, for the question "Who is JJ Lin's wife?", with the standard answer "Ding Wenqi", model predictions like "JJ Lin's wife", "JJ Lin's wife should be excellent", "JJ Lin's wife might be a public figure" are all [INCORRECT].
- If the standard answer contains more information than the question asks for, the predicted answer only needs to include the information mentioned in the question.
    - For example, for the question "What is the main chemical component of magnesite?", with the standard answer "Magnesium carbonate (MgCO3)", "Magnesium carbonate" or "MgCO3" are both considered [CORRECT] answers.
- If information omitted in the predicted answer can be clearly inferred from the question, it's considered correct.
    - For example, for the question "The Nuragic ruins of Barumini were listed as a World Cultural Heritage by UNESCO in 1997, so where is this site located?", with the standard answer "Sardinia, Italy", the predicted answer "Sardinia" is considered [CORRECT].
- If it's clear that different translations of a name refer to the same person, it's considered correct.
    - For example, if the standard answer is "Robinson", answers like "Lubinson" or "Lubinsun" are both correct.
- You should focus more on the match between the standard answer and the model prediction, rather than whether the standard answer itself is correct.

Below is a new question example. Please reply with only [CORRECT] or [INCORRECT], without apologies or corrections to your own errors, just evaluate the answer.
```
Question: {question}
Standard Answer: {correct_answer}
Predicted Answer: {response}
```

Evaluate this new question's predicted answer as one of the following:
A. [CORRECT]
B. [INCORRECT]

Return only the option representing [CORRECT] or [INCORRECT], i.e. just return A or B, without adding any other text.
""".strip()


SYSTEM_PROMPT_DEEPSEEK_V32 = """You are a deep research assistant. Your primary role is to conduct rigorous, multi-source investigations on any topic. You must be capable of handling both broad, open-domain questions and inquiries within specialized academic or technical fields.

**Conduct Explicit Planning**
When solving complex deep search problem, you should explicitly plan your reasoning and information acquisition process.

**Maintain a Structured Research Notebook**
When facing non-trivial questions, you should explicitly maintain a structured research notebook to record your plans, reasoning, key information obtained, and information that requires further exploration.
Always be explicit about:
- Confirmed information
    1. What has already been verified
    2. What evidence supports it
- Missing information / gaps
    1. Which constraints are still unresolved
    2. What is unknown or uncertain
- Priority gaps
    1. Which missing information most reduces uncertainty if resolved next 
    2. Why resolving it will move you closer to the answer
Think of this as keeping a small but explicit “notebook” while you work.

**Explicit Reasoning before Each Tool Call**
Before performing any external search or tool call, explain your intent throught reasoning:
- What information is currently missing or uncertain
- What the goal of this tool call is and What information you expect to obtain from it.

**Periodic Progress Check**
Complex problems require course correction. 
Every 5～10 reasoning steps, you should perform a brief “progress check” and update your ”research notebook”:
1. Summarizes what has been confirmed so far (with key evidence)
2. Restates the remaining open gaps and their priority order
3. Plans the next few steps (typically 2–4), each mapped to a specific gap
4. Notes any hypothesis that is currently guiding the search, and what would falsify it
Its purpose is to prevent drift and ensure each next action reduces the most important uncertainty.


**Output Format**:
After collecting all necessary information, provide two outputs:
1. A structured summary that presents your reasoning process and compiles all information necessary to formulate the final answer.
2. A final answer that is clear, direct, and well-supported by the synthesized evidence.
Your output should follow the following format:

<summary>
{your summary}

<answer>
{your answer}
"""

TOOLS_DEEPSEEK_V32 = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Searches for information related to query and displays topn results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string"
                    },
                    "topn": {
                        "type": "integer",
                        "description": "Number of top results to display",
                        "default": 10
                    },
                    "source": {
                        "type": "string",
                        "description": "Source to search within",
                        "enum": [
                            "web",
                            "news"
                        ],
                        "default": "web"
                    }
                },
                "required": [
                    "query"
                ],
                "additionalProperties": "false",
                "$schema": "http://json-schema.org/draft-07/schema#"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "visit",
            "description": "Visit one or more webpages and return a summary of their content tailored to the specified goal. Returns separate summaries for each URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "A list of webpage URLs to visit."
                    },
                    "goal": {
                        "type": "string",
                        "description": "The specific information to extract or focus on when summarizing the webpage content."
                    }
                },
                "required": ["url", "goal"]
            }
        }
    },
    {
        "type": "function",
            "function": {
            "name": "python",
            "description": "A utility that executes Python 3.11 code. Users must explicitly import any required libraries. The tool runs the provided Python code and returns both stdout and stderr. You should print results explicitly to ensure they appear in the returned output.",
            "parameters": {
                "type": "object",
                "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python code to be executed."
                }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
        "name": "google_scholar",
        "description": "Leverage Google Scholar to retrieve relevant information from academic publications. Accepts multiple queries.",
        "parameters": {
            "type": "object",
            "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "The search query."
                },
                "minItems": 1,
                "description": "The list of search queries for Google Scholar."
            }
            },
            "required": ["query"],
            "additionalProperties": "false",
            "$schema": "http://json-schema.org/draft-07/schema#"
        }
    }
  },
  {
        "type": "function",
        "function": {
        "name": "google_maps",
        "description": "Search Google Maps places. Returns a list of places with name, address, coordinates, ratings, categories, opening hours, and place identifiers.",
        "parameters": {
            "type": "object",
            "properties": {
            "q": {
                "type": "string",
                "description": "Google Maps search query."
            },
            "page": {
                "type": "integer",
                "description": "Page number of results.",
                "default": 1,
                "minimum": 1
            }
            },
            "required": ["q"],
            "additionalProperties": "false",
            "$schema": "http://json-schema.org/draft-07/schema#"
      }
    }
  }
]