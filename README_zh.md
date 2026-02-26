# DeepTraceHub

DeepTraceHub 是 [REDSearcher](https://github.com/RedSearchAgent/REDSearcher) 使用的一个模块化且可扩展的框架，用于构建和评估基于大语言模型（LLM）的智能体（Agents），专注于深度推理和深度信息检索。它旨在支持智能体开发的完整生命周期，包括 **搜索智能体评估**、用于模型训练的 **轨迹合成** 以及 **多阶段问题校验**。

它支持多种智能体循环（如 ReACT 和 DeepSeek 风格的思考过程），并提供了一系列搜索和检索工具。

## 核心特性

- **模块化智能体架构**：轻松在不同的智能体循环（ReACT、Thinking-with-Tools 等）之间切换。
- **可扩展的工具系统**：内置对 Google 搜索（通过 Serper）、Jina 以及本地搜索/爬取工具的支持。
- **高性能**：支持多进程和异步执行，适用于大规模推理和评估。
- **内置评估系统**：集成了多个裁判模型（GAIA、XBench 等），用于自动化性能衡量。
- **负载均衡**：原生支持 LLM 端点的负载均衡（SGLang、vLLM）。

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 配置

Deeptracehub 使用 YAML 配置文件来定义智能体行为、LLM 端点和工具设置。示例请参考 `config/config.yaml`（或类似文件）。

### 2. 部署模型

在运行智能体之前，您需要部署必要的 LLM 服务。Deeptracehub 兼容 OpenAI 风格的 API（例如通过 vLLM 或 SGLang 部署）。

#### 2.1 智能体模型 (Agent Model)
核心推理模型（例如 REDSearcher、DeepSeek 等）。它负责处理主要的 ReACT 智能体循环。
- **推荐**：使用 vLLM 或 SGLang 以实现高吞吐量推理。
- **配置**：设置 `llm.local.endpoint_list` 和 `llm.local.model_name`。

#### 2.2 摘要模型 (Summarizer Model)
一个更小、更快的模型（例如 Qwen3-30B-A3B-Instruct），专门用于总结爬虫工具获取的网页内容。
- **配置**：设置 `tools.summarize.endpoint` 和 `tools.summarize.model_name`。

#### 2.3 裁判模型 (LLM-as-Judge Model)
一个强大的模型（例如 GPT-OSS-120B），用于根据标准答案评估智能体的最终回答。
- **配置**：设置 `judge.local.endpoint` 和 `judge.local.model_name`。

### 3. 运行推理

运行推理的主要入口是 `src/agent.py`。您可以使用配置文件运行它：

```bash
mkdir /path/to/output_dir
export LOG_FILE=/path/to/output_dir/logs.log
python src/agent.py --config_path config/your_config.yaml --multiprocess --resume
```

**关键参数：**
- `--config_path`：YAML 配置文件的路径。
- `--multiprocess`：启用多进程执行（默认：True）。
- `--resume`：如果执行中断，从之前的检查点恢复。
- `--debug`：以调试模式运行（单线程）。

### 4. 输出结果

结果保存在配置文件中指定的 `output_dir` 目录下，通常为 JSON 格式，包括：
- `problem_idx.json`：每个查询的详细轨迹和最终答案。
- `results.json`：聚合统计数据和所有结果。
- `accuracy.json`：简明的准确率摘要。

## 配置详解

### 1. API 密钥管理 (.env)

Deeptracehub 使用 `.env` 文件来管理敏感的 API 密钥。在根目录下创建一个 `.env` 文件：

```env
## python 代码沙箱
CODE_SANDBOX_URL=""
CODE_SANDBOX_TOKEN=""

## serper.dev API 密钥 (用于 google search/maps/scholar 和爬虫备份)
SERPERDEV_TOKEN=""
## jina API 密钥 (用于网页爬取)
JINA_TOKEN=""

## Gemini API 密钥
GEMINI_TOKEN=""

## 本地搜索和爬取
# serve_wikipedia.py
LOCAL_SEARCH_URL=""
# serve_wikidata.py
LOCAL_CRAWL_URL="" 
```

### 2. config.yaml 参数含义

配置分为以下几个部分：

#### `agent`
- `cls`：要使用的智能体类（例如 `ReACTAgent`、`DeepSeekV32ThinkingWithToolAgent`）。
- `max_iter`：最大交互轮次（Thought-Action-Observation）。
- `model`：用于选择分词器（tokenizer）和系统提示词的模型标识符（例如 `redsearcher`）。
- `sampling.enable`：是否为每个输入查询采样多条轨迹。
- `sampling.n_samples`：采样数量。

#### `exps` (实验设置)
- `input_file`：输入 JSONL 文件的路径（包含 `idx`, `problem`, `answer`）。
- `input_file_list`：输入文件列表。
- `output_dir`：保存结果和日志的目录。
- `num_processes`：多进程模式下启动的进程数。
- `max_workers`：总并发数，等于 min(`num_processes` * `threads_per_process`, `max_workers`)。
- `threads_per_process`：每个进程内的异步线程数。
- `queue_multiplier`：控制活跃任务缓冲区（`threads * multiplier`），以优化 LLM 后端的 Prefix Cache 命中率。

#### `llm`
- `local.endpoint`：LLM 的主要 API 端点。
- `local.endpoint_list`：用于自动负载均衡的端点列表（推荐）。
- `local.model_name`：OpenAI API 要求的模型名称字符串。

#### `judge`
- `remote`：默认为 False。是否使用远程裁判（如 Gemini）或本地 LLM。
- `local.endpoint`：本地裁判模型的 API 端点。
- `local.model_name`：本地裁判模型的名称（例如 `gpt-oss`）。

#### `tools`
- `search_tool`：首选搜索提供商（`serper.dev`, `local`, `empty`）。
- `crawl_tool`：首选爬虫提供商（`jina`, `serper.dev`, `local`）。
- `crawl_tool_backup`：主爬虫失败时的备用爬虫提供商。
- `summarize_tool`：使用的基于 LLM 的摘要工具（例如 `brief_llm_summarizer`）。
- `serper_key_usage_file`：可选的 JSON 文件路径，用于跟踪 Serper API 密钥的使用情况。

- `search` (搜索设置)：
  - `cache_dir`：存储搜索结果缓存的路径。
  - `num`：每个查询检索的搜索结果数量。
  - `timeout`：请求超时时间（秒）。
  - `retry.max_attempts`：搜索请求的最大重试次数。

- `crawl` (爬虫设置)：
  - `cache_dir`：存储爬取页面缓存的路径。
  - `jina.token_budget`：Jina 爬取的 token 预算（-1 表示无限制）。
  - `timeout`：请求超时时间（秒）。

- `summarize` (摘要设置)：
  - `endpoint`：用于总结网页内容的 LLM API 端点。
  - `model_name`：摘要任务使用的模型名称。
  - `cache_dir`：存储摘要缓存的路径。

- `python` (Python 沙箱设置)：
  - `run_timeout`：Python 代码沙箱的最大执行时间。
  - `compile_timeout`：最大编译时间。

## 自定义开发

### 1. 添加新的搜索/爬虫工具

要添加新的搜索或爬虫工具：

1.  在 `tools/search.py` 中 **实现工具类**：
    - 对于搜索工具，继承 `SearchToolBase` 并实现 `search(self, query: str)`。
    - 对于爬虫工具，继承 `WebCrawlToolBase` 并实现 `crawl(self, url: str)`。

    ```python
    class MyNewSearchTool(SearchToolBase):
        name = "my_new_tool"
        async def search(self, query: str) -> OrganicResults:
            # 在此处实现您的逻辑
            pass
    ```

2.  在 `src/agents/base.py` 中 **注册工具**：
    - 将您的类添加到 `TOOL_CLASS_SEARCH` 或 `TOOL_CLASS_CRAWL` 中。

### 2. 添加新的智能体循环

要实现自定义的智能体逻辑：

1.  在 `src/agents/` 中 **创建一个新的智能体类**：
    - 继承 `AgentBase`。
    - 实现 `process_query` 异步方法。

    ```python
    from src.agents.base import AgentBase

    class MyCustomAgent(AgentBase):
        async def process_query(self, idx, user_query, **kwargs):
            # 在此处定义您的自定义推理循环
            # 使用 self.search_tool, self.web_crawl_tool 等。
            pass
    ```

2.  在 `src/agents/__init__.py` 中 **导出智能体**：
    - 导入您的类并将其添加到 `__all__`。

3.  **更新您的配置**：
    - 将 `agent.cls` 设置为您的新类名（例如 `MyCustomAgent`）。
    
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