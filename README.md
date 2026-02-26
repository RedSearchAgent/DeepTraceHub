# DeepTraceHub

DeepTraceHub is a modular and extensible framework for building and evaluating LLM-based Agents, with a focus on deep reasoning and deep information seeking used in [REDSearcher](https://github.com/RedSearchAgent/REDSearcher). It is designed to support the full lifecycle of agent development, including **search agent evaluation**, **trajectory synthesis** for model training, and **multi-stage problem verification**. 
It supports various agent loops (like ReACT and DeepSeek-style thinking) and provides a suite of search and retrieval tools.

## Key Features

- **Modular Agent Architecture**: Easily switch between different agent loops (ReACT, Thinking-with-Tools, etc.).
- **Extensible Tool System**: Built-in support for Google Search (via Serper), Jina, and local search/crawl tools.
- **High Performance**: Supports multi-process and asynchronous execution for large-scale inference and evaluation.
- **Built-in Evaluation**: Integrated with multiple judges (GAIA, XBench, etc.) for automated performance measurement.
- **Load Balancing**: Native support for LLM endpoint load balancing (SGLang, vLLM).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Configuration

Deeptracehub uses YAML configuration files to define agent behavior, LLM endpoints, and tool settings. See `config/config.yaml` (or similar) for examples.

### 2. Deploy Models

Before running the agent, you need to deploy the necessary LLM services. Deeptracehub is compatible with OpenAI-style APIs (e.g., deployed via vLLM or SGLang).

#### 2.1 Agent Model
The core reasoning model (e.g., REDSearcher, DeepSeek, etc). It handles the main ReACT Agent loops.
- **Recommended**: vLLM or SGLang for high-throughput inference.
- **Config**: Set `llm.local.endpoint_list` and `llm.local.model_name`.

#### 2.2 Summarizer Model
A smaller, faster model (e.g., Qwen3-30B-A3B-Instruct) used specifically to summarize web page content retrieved by crawl tools.
- **Config**: Set `tools.summarize.endpoint` and `tools.summarize.model_name`.

#### 2.3 LLM-as-Judge Model
A strong model (e.g., GPT-OSS-120B) used to evaluate the agent's final answers against ground truth.
- **Config**: Set `judge.local.endpoint` and `judge.local.model_name`.

### 3. Running Inference

The main entry point for running inference is `src/agent.py`. You can run it with a configuration file:

```bash
mkdir /path/to/output_dir
export LOG_FILE=/path/to/output_dir/logs.log
python src/agent.py --config_path config/your_config.yaml --multiprocess --resume
```

**Key Arguments:**
- `--config_path`: Path to your YAML configuration.
- `--multiprocess`: Enable multi-process execution (default: True).
- `--resume`: Resume from previous checkpoints if execution was interrupted.
- `--debug`: Run in debug mode (single-threaded).

### 3. Output

Results are saved in the `output_dir` specified in your config file, typically in JSON format, including:
- `problem_idx.json`: Detailed trajectory and final answer for each query.
- `results.json`: Aggregated statistics and all results.
- `accuracy.json`: Concise accuracy summary.

## Configuration

### 1. API Key Management (.env)

Deeptracehub uses a `.env` file to manage sensitive API keys. Create a `.env` file in the root directory:

```
## python code sandbox
CODE_SANDBOX_URL=""
CODE_SANDBOX_TOKEN=""

## serper.dev API key (for google search/maps/scholar and crawl backup)
SERPERDEV_TOKEN=""
## jina API key (for web crawl)
JINA_TOKEN=""

## Gemini API key
GEMINI_TOKEN=""

## Local Search and Crawl
# serve_wikipedia.py
LOCAL_SEARCH_URL=""
# serve_wikidata.py
LOCAL_CRAWL_URL="" 
```

### 2. Parameter Meanings in config.yaml

The configuration is divided into several sections:

#### `agent`
- `cls`: The Agent class to use (e.g., `ReACTAgent`, `DeepSeekV32ThinkingWithToolAgent`).
- `max_iter`: Maximum number of interaction rounds (Thought-Action-Observation).
- `model`: Model identifier used for selecting tokenizer and system prompts (e.g. `redsearcher`).
- `sampling.enable`: Enable sampling multiple trajs for each input query.
- `sampling.n_smaples`: The number of samples.

#### `exps` (Experiments)
- `input_file`: Path to the input JSONL file (contains `idx`, `problem`, `answer`).
- `input_file_list`: List of input_file.
- `output_dir`: Directory where results and logs will be saved.
- `num_processes`: Number of parallel processes to spawn with multi-process. 
- `max_workers`: The overall concurrency equals min(`num_processes` times `threads_per_process`, `max_workers`)
- `threads_per_process`: Number of async threads within each process.
- `queue_multiplier`: Controls the active task buffer (`threads * multiplier`) to optimize prefix cache hits on the LLM backend.


#### `llm`
- `local.endpoint`: The primary API endpoint for the LLM.
- `local.endpoint_list`: A list of endpoints for automatic load balancing (recommended).
- `local.model_name`: The model name string required by the OpenAI API.

#### `judge`
- `remote`: Default to False. Whether to use a remote judge (e.g., Gemini) or a local LLM.
- `local.endpoint`: The API endpoint for the local judge model.
- `local.model_name`: The model name for the local judge (e.g., `gpt-oss`).

#### `tools`
- `search_tool`: The primary search provider (`serper.dev`, `local`, `empty`).
- `crawl_tool`: The primary crawl provider (`jina`, `serper.dev`, `local`).
- `crawl_tool_backup`: A backup crawl provider if the primary one fails.
- `summarize_tool`: The LLM-based summarizer to use (e.g., `brief_llm_summarizer`).
- `serper_key_usage_file`: Optional path to a JSON file for tracking Serper API key usage.

- `search`:
  - `cache_dir`: Path to store search result caches.
  - `num`: Number of search results to retrieve per query.
  - `timeout`: Request timeout in seconds.
  - `retry.max_attempts`: Maximum number of retries for search requests.

- `crawl`:
  - `cache_dir`: Path to store crawled page caches.
  - `jina.token_budget`: Token limit for Jina crawl (-1 for unlimited).
  - `timeout`: Request timeout in seconds.

- `summarize`:
  - `endpoint`: LLM API endpoint for summarizing web pages.
  - `model_name`: Model name for the summarization task.
  - `cache_dir`: Path to store summary caches.

- `python`:
  - `run_timeout`: Maximum execution time for the Python code sandbox.
  - `compile_timeout`: Maximum compilation time.

## Customization

### 1. Adding New Search/Crawl Tools

To add a new search or crawl tool:

1.  **Implement the tool class** in `tools/search.py`:
    - For search tools, inherit from `SearchToolBase` and implement `search(self, query: str)`.
    - For crawl tools, inherit from `WebCrawlToolBase` and implement `crawl(self, url: str)`.

    ```python
    class MyNewSearchTool(SearchToolBase):
        name = "my_new_tool"
        async def search(self, query: str) -> OrganicResults:
            # Your implementation here
            pass
    ```

2.  **Register the tool** in `src/agents/base.py`:
    - Add your class to `TOOL_CLASS_SEARCH` or `TOOL_CLASS_CRAWL`.

### 2. Adding New Agent Loops

To implement a custom agent logic:

1.  **Create a new agent class** in `src/agents/`:
    - Inherit from `AgentBase`.
    - Implement the `process_query` asynchronous method.

    ```python
    from src.agents.base import AgentBase

    class MyCustomAgent(AgentBase):
        async def process_query(self, idx, user_query, **kwargs):
            # Define your custom reasoning loop here
            # Use self.search_tool, self.web_crawl_tool, etc.
            pass
    ```

2.  **Export the agent** in `src/agents/__init__.py`:
    - Import your class and add it to `__all__`.

3.  **Update your config**:
    - Set `agent.cls` to your new class name (e.g., `MyCustomAgent`).
  
## Citation

```
@article{redsearcher2026,
  title={REDSearcher: A Scalable and Cost-Efficient Framework for Long-Horizon Search Agents},
  author={Zheng Chu and Xiao Wang and Jack Hong and Huiming Fan and Yuqi Huang and Yue Yang and Guohai Xu and Shengchao Hu and Dongdong Kuang and Chenxiao Zhao and Cheng Xiang and Ming Liu and Bing Qin and Xing Yu},
  journal={arXiv preprint arXiv:2602.14234},
  url={https://arxiv.org/pdf/2602.14234},
  year={2026}
}
```

