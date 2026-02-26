"""
Agents module: Contains all Agent classes.

This module provides a unified entry point for all Agent classes.
"""

# Import base class and tools
from src.agents.base import (
    AgentBase,
    MODEL_ARGS,
    log_exception_details,
    TOOL_MAP_SEARCH,
    TOOL_MAP_CRAWL,
    TOOL_MAP_SUMMARIZER,
)

# Import Agent classes
from src.agents.direct_qa import DirectQAVerifier
from src.agents.react import ReACTAgent
from src.agents.dsv32 import DeepSeekV32ThinkingWithToolAgent

# Export all classes
__all__ = [
    # Base class
    "AgentBase",
    "MODEL_ARGS",
    "log_exception_details",
    "TOOL_MAP_SEARCH",
    "TOOL_MAP_CRAWL", 
    "TOOL_MAP_SUMMARIZER",
    # Agent classes
    "DirectQAVerifier",
    "ReACTAgent",
    "DeepSeekV32ThinkingWithToolAgent",
]
