import asyncio
import random
import time
import os
from typing import Optional, Dict
from openai import AsyncOpenAI, OpenAI
import logging

from llms import execute_with_retries

logger = logging.getLogger(__name__)

# Global client cache to avoid duplicate creation
_client_cache: Dict[str, AsyncOpenAI] = {}
_sync_client_cache: Dict[str, OpenAI] = {}

def get_or_create_client(endpoint: str) -> AsyncOpenAI:
    """Get or create client, use cache to avoid duplicate creation"""
    api_key = os.getenv("DIRECT_LLM_API_KEY", "abc")
    if endpoint not in _client_cache:
        _client_cache[endpoint] = AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key
        )
    return _client_cache[endpoint]

def sync_get_or_create_client(endpoint: str) -> OpenAI:
    if endpoint not in _sync_client_cache:
        _sync_client_cache[endpoint] = OpenAI(
            base_url=endpoint,
            api_key="xxxxx"
        )
    return _sync_client_cache[endpoint]

async def cleanup_clients():
    """Clean up all client connections"""
    for client in _client_cache.values():
        await client.close()
    _client_cache.clear()

async def execute_with_retries_async(
    operation,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    context: str = "operation",
) -> any:
    """Run `operation` with retry and exponential backoff using async sleep."""
    attempt = 0
    current_delay = delay
    last_exc: Optional[Exception] = None

    while attempt < max_attempts:
        try:
            return await operation()
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
            await asyncio.sleep(max(current_delay, 0) * jitter)
            current_delay *= backoff_factor

    if last_exc is not None:
        logger.error(f"LLM call failed after {max_attempts} attempts for {context}: {last_exc}")
        raise last_exc

    raise RuntimeError(f"LLM call failed after {max_attempts} attempts for {context}")

async def async_completion(prompt: str, model, endpoint=None):
    extra_body = {
        "skip_special_tokens": False
    }
    if "gpt-oss" in model.lower():
        model = "gpt-oss"
    
    if endpoint is not None:
        client_local = get_or_create_client(endpoint)
        async def _make_request():
            return await client_local.completions.create(
                model=model,
                prompt=prompt,
                max_tokens=8192,
                temperature=1.0,
                top_p=0.95,
                n=1,
                extra_body=extra_body,
            )
        resoponse = await execute_with_retries_async(
            _make_request,
            context=f"async_completion for model {model}"
        )
        return resoponse.choices[0]
    else:
        logger.error(f"LLM endpoint is not provided: {endpoint}")
        raise RuntimeError("LLM endpoint is not provided.")

async def async_chat_completion_gpt_oss(messages, model, tools, endpoint=None):
    extra_body = {
        "repetition_penalty": 1.1
    }
    if "deepseek" in model.lower() and ("v3.2" in model.lower() or "v32" in model.lower() or "v3_2" in model.lower() or "v3-2" in model.lower()):
        extra_body.update({
            "chat_template_kwargs": {"thinking": True},
            "separate_reasoning": True
        })
    if "low" in model.lower():
        reasoning_effort = "low"
    elif "high" in model.lower():
        reasoning_effort = "high"
    else:
        reasoning_effort = "medium"
    if "gpt-oss" in model.lower():
        model = "gpt-oss"

    if endpoint is not None:
        client_local = get_or_create_client(endpoint)

        async def _make_request():
            return await client_local.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                n=1,
                temperature=0.85,
                top_p=0.95,
                tools=tools,
                tool_choice="auto",
                extra_body=extra_body,
                reasoning_effort=reasoning_effort,
                max_tokens=8192
            )
        resoponse = await execute_with_retries_async(
            _make_request,
            context=f"async_chat_completion_gpt_oss for model {model}"
        )
        return resoponse.choices[0].message
    else:
        logger.error(f"LLM endpoint is not provided: {endpoint}")
        raise RuntimeError("LLM endpoint is not provided.")

def sync_chat_completion(messages, model, endpoint=None):
    """同步版本的 chat_completion"""
    extra_body = {
        "repetition_penalty": 1.1
    }
    if "deepseek" in model.lower() and ("v3.2" in model.lower() or "v32" in model.lower() or "v3_2" in model.lower() or "v3-2" in model.lower()):
        extra_body.update({
            "chat_template_kwargs": {"thinking": True},
            "separate_reasoning": True
        })
    if endpoint is not None:
        client_local = sync_get_or_create_client(endpoint)
        def _make_request():
            return client_local.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                n=1,
                temperature=0.85,
                top_p=0.95,
                extra_body=extra_body,
                max_tokens=8192
            )
        response = execute_with_retries(
            _make_request,
            context=f"sync_chat_completion for model {model}"
        )
        return response.choices[0].message.content
    else:
        logger.error(f"LLM endpoint is not provided: {endpoint}")
        raise RuntimeError("LLM endpoint is not provided.")

async def async_chat_completion(messages, model, endpoint=None, n=1, temperature=0.85):
    """异步版本的 chat_completion，使用缓存的客户端"""
    extra_body = {
        "repetition_penalty": 1.1
    }

    if endpoint is not None:
        client_local = get_or_create_client(endpoint)
        
        async def _make_request():
            return await client_local.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                n=n,
                temperature=temperature,
                top_p=0.95,
                # Can add repetition_penalty parameter. Model always repeats during generation, causing long tail, which seriously damages rollouts.
                extra_body=extra_body,
                max_tokens=8192
            )
        
        response = await execute_with_retries_async(
            _make_request,
            context=f"async_chat_completion for model {model}"
        )
        if n == 1:
            return response.choices[0].message.content
        else:
            return [choice.message.content for choice in response.choices]
    else:
        logger.error(f"LLM endpoint is not provided: {endpoint}")
        raise RuntimeError("LLM endpoint is not provided.")

# For compatibility, can also provide a sync wrapper
def chat_completion_sync_wrapper(messages, model, endpoint=None):
    """Sync wrapper, internally uses async implementation"""
    return asyncio.run(async_chat_completion(messages, model, endpoint))
