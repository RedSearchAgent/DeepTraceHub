import os
import sys
import random
import time
import logging
from typing import Callable, Optional, TypeVar
import requests
from openai import OpenAI

# Set up logging
logger = logging.getLogger(__name__)

# Type variable
T = TypeVar('T')

if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2])
    sys.path.append(base_path)
    
from utils.file_utils import FileUtils

GEMINI_API_KEY = os.getenv("GEMINI_TOKEN")
class GeminiClient:
    """Gemini API client"""
    
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        self.url = "GEMINI_URL"
        self.headers = {
            'api-key': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def generate_content(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096, timeout: int = 75) -> str:
        """Call Gemini API to generate content"""
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": 0.8,
                "seed": random.randint(0, 1000)
            }
        }
        max_trials = 3
        for trial in range(max_trials):
        
            try:
                response = requests.post(self.url, headers=self.headers, json=payload, timeout=timeout)
                response.raise_for_status()
                
                response_json = response.json()
                if "candidates" in response_json and len(response_json["candidates"]) > 0:
                    content = response_json["candidates"][0]["content"]["parts"][0]["text"]
                    return content.strip()
                else:
                    raise RuntimeError(f"[Gemini LLM as Judge] API response doesn't contain candidates.")
                    
            except requests.exceptions.RequestException as e:
                print(f"API request failed: {e}")
                continue
            except (KeyError, IndexError) as e:
                print(f"Failed to parse API response: {e}")
                continue
            except Exception as e:
                print(f"Unknown error when calling API: {e}")
                continue
            
        return "llm as judge failed"

gemini_client = GeminiClient()

def execute_with_retries(
    operation: Callable[[], T],
    *,
    max_attempts: int = 7,
    delay: float = 3,
    backoff_factor: float = 2.0,
    context: str = "operation",
) -> T:
    """Run `operation` with retry and exponential backoff using sync sleep."""
    attempt = 0
    current_delay = delay
    last_exc: Optional[Exception] = None

    while attempt < max_attempts:
        try:
            return operation()
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
            time.sleep(max(current_delay, 0) * jitter)
            current_delay *= backoff_factor

    if last_exc is not None:
        logger.error(f"LLM call failed after {max_attempts} attempts for {context}: {last_exc}")
        raise last_exc

    raise RuntimeError(f"LLM call failed after {max_attempts} attempts for {context}")

# temperature and top-p are set to 0.85 and 0.95 from WebSailor-v2
def chat_completion(messages, model, endpoint=None):
    if endpoint is not None:
        client_local = OpenAI(
            base_url=endpoint,
            api_key="abc"
        )
    else:
        logger.error(f"LLM endpoint is not provided: {endpoint}")
        raise RuntimeError("LLM endpoint is not provided.")
    
    def _make_request():
        return client_local.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,
            n=1,
            temperature=0.85,
            top_p=0.95,
            extra_body={
                "chat_template_kwargs": {
                    # "thinking": True,
                    # "enable_thinking": True,
                    # "add_generation_prompt": True,
                }
            }
        )
    
    response = execute_with_retries(
        _make_request,
        context=f"chat_completion for model {model}"
    )
    return response.choices[0].message.content

def chat_completion_gemini(prompt: str) -> str:
    response = gemini_client.generate_content(prompt)
    return response
