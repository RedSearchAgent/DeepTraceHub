"""
Judge Module
Provides unified evaluation interface and factory function
"""

from .base_judge import BaseJudge
from .gaia_judge import GAIAJudge
from .xbench_judge import XBenchJudge


def get_judge(data_source, remote, model_name, endpoint) -> BaseJudge:
    """
    Get corresponding judge based on data source
    
    Args:
        data_source: Data source identifier (filename, e.g., "gaia_200", "xbench_test", etc.)
        remote: Whether to use remote evaluation
        model_name: Model name
        endpoint: Evaluation endpoint
    
    Returns:
        BaseJudge: Corresponding judge instance
    """
    # Handle null case
    if data_source is None:
        return BaseJudge(remote, model_name, endpoint)
    
    data_source_lower = data_source.lower()
    
    # GAIA task
    if "gaia" in data_source_lower:
        print(f" [INFO] Data source '{data_source}' uses GAIA judge")
        return GAIAJudge(remote, model_name, endpoint)
    
    # XBench task
    elif "xbench" in data_source_lower:
        print(f" [INFO] Data source '{data_source}' uses XBench judge")
        return XBenchJudge(remote, model_name, endpoint)
    
    # BrowseComp task (including browsecomp, browsecomp-zh, browsecomp-en, etc.)
    elif "browse_comp" in data_source_lower or "hle" in data_source_lower:
        print(f" [INFO] Data source '{data_source}' uses BrowseComp judge")
        return BaseJudge(remote, model_name, endpoint)
    
    # Other tasks use default judge
    else:
        # print(f" [INFO] Data source '{data_source}' uses default judge")
        print(f" [INFO] Data source '{data_source}' uses default judge")
        return BaseJudge(remote, model_name, endpoint)


__all__ = [
    'BaseJudge',
    'GAIAJudge',
    'XBenchJudge',
    'get_judge'
]

