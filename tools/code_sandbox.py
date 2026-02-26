
import os
import re
import sys
import json
import httpx
import asyncio
import traceback
from typing import Dict, Any

CODE_SANDBOX_URL = os.getenv("CODE_SANDBOX_URL")

if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2])
    sys.path.append(base_path)

from utils.logger import get_logger
from tools.search import execute_with_retries

logger = get_logger(__name__)

async def invoke_code_sandbox(payload: dict):
    """Use httpx to asynchronously call the new code sandbox interface"""
    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.post(
                CODE_SANDBOX_URL,
                json=payload,
                timeout=240.0,  # Set a longer timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Invoke code sandbox failed: {e}")
            raise RuntimeError(f"Invoke code sandbox failed: {e}")

async def run_python(code: str, **kwargs) -> str:
    """Run Python code and return formatted output"""
    payload = {
        "code": code,
        "language": "python",
        "compile_timeout": kwargs.get("compile_timeout", 60),
        "run_timeout": kwargs.get("run_timeout", 120),
    }

    async def _run_python() -> Dict[str, Any]:
        try:
            result = await invoke_code_sandbox(payload=payload)
            return result
        except Exception as e:
            raise RuntimeError(f"Python execution failed: {e}") from e

    try:
        # Use retry mechanism
        result = await execute_with_retries(
            _run_python,
            context=f"Python execution for code: {code[:100]}...",
            max_attempts=kwargs.get("retry", {}).get("max_attempts", 2),
            delay=kwargs.get("retry", {}).get("delay", 2.0),
            backoff_factor=kwargs.get("retry", {}).get("backoff_factor", 2.0),
        )

        run_result = result.get("run_result", {})
        stdout, stderr = run_result.get("stdout", ""), run_result.get("stderr", "")
        
        result_list = []
        if stdout:
            result_list.append(f"stdout:\n{stdout}")
        if stderr:
            result_list.append(f"stderr:\n{stderr}")
        if run_result.get("status") == "TimeLimitExceeded":
            result_list.append("[Python Interpreter Error] TimeoutError: Execution timed out.")
        
        return "\n".join(result_list).strip()

    except Exception as e:
        logger.error(f"Python execution failed: {e}")
        return "[Python Interpreter Error]: All attempts failed."



if __name__ == "__main__":
    test1 = """a = 4
b = 5
result = a + b
print("a + b = ", result)"""

    test2 = """import time
time.sleep(3)
print("Success")"""

    result1 = asyncio.run(run_python(code=test1))
    result2 = asyncio.run(run_python(code=test2))

    print(result1)
    print(result2)
