import asyncio
import json
import os
import sys
import random
import logging
import argparse
import traceback
import itertools
import multiprocessing as mp
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
from openai import AsyncOpenAI
from multiprocessing import Pool

# Add base path to sys.path
if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2])
    sys.path.append(base_path)

from utils.file_utils import FileUtils, logger
logger = logging.getLogger(__name__)

# ==========================================
# Validator Prompt Template
# ==========================================
WEBQA_VALIDATOR_PROMPT = """
You are a rigorous data quality auditor for a Deep Search dataset. Your task is to verify whether the **Fuzzed Query** is *legal* given the provided evidence, and whether the **Expected Answer** still satisfies the query.

### TASK INSTRUCTIONS

#### A) Evidence Scope & Hierarchy
When validating the Fuzzed Query, you must use **all** of the following as evidence sources:

1) **Strong evidence** (highest priority, ground truth):
- `context_text`
- `node_details`
- `involved_relations`

2) **Supporting evidence**:
- `entity_replace_context` (paraphrased/obfuscated facts; may omit names but should preserve meaning)

**Critical rule (default non-hallucination):**
- The Fuzzed Query may add new details/constraints not present in the evidence.
- Treat any such new detail as **valid by default** unless you can show one of the following:
  (i) it **explicitly contradicts** strong evidence, OR
  (ii) it makes the expected answer **definitively impossible** to satisfy (definitive answer-breaking), OR
  (iii) you are **highly confident** it is objectively false (clear real-world contradiction).
- If a statement is unsupported by strong evidence, you must label it **Unverifiable but non-contradictory** (NOT hallucination), unless (i)-(iii) applies.

If any statement in the Fuzzed Query **explicitly contradicts** Strong evidence, treat it as a serious inconsistency.

The Fuzzed Query is allowed to:
- relax constraints (become less specific),
- paraphrase/obfuscate entities/events,
- introduce extra constraints if they are (a) consistent with strong evidence, OR (b) unverifiable but non-contradictory and do not break the answer.

The Fuzzed Query is **illegal** if it:
- contradicts strong evidence, OR
- introduces new constraints that **definitively** make the expected answer not satisfy the query (“definitive answer-breaking”), OR
- makes the target ambiguous or type-mismatched.

---

#### B) Validation Steps (produce reasoning for each)
Before the final verdict, you must produce reasoning for each dimension:

##### 1) Hallucination / Inconsistency Check (Core)
1. Decompose the Fuzzed Query into **atomic constraints** (A, B, C...).
2. For each constraint, label it as one of:
   - **Supported** (explicitly in strong evidence)
   - **Unverifiable but non-contradictory** (no support found, no conflict found; default pass)
   - **Contradictory** (explicitly conflicts with strong evidence or clearly false in the real world)
   - **Definitive answer-breaking** (given strong evidence, the expected answer **cannot** satisfy the constraint)
3. Conclude:
   - `hallucination_ok = true` iff there is **no** constraint labeled **Contradictory** or **Definitive answer-breaking**.
   - If constraints are only “Unverifiable but non-contradictory”, set `hallucination_ok=true` but explicitly note the uncertainty risk.

**Numeric tolerance rule (important):**
- If strong evidence provides a specific number `x`, and fuzzed_query uses a range/approximation that **includes `x`**
  (e.g., “about x”, “x±k”, “between a and b” where `a ≤ x ≤ b`), label it as **Supported** (or at least Unverifiable-but-non-contradictory), NOT contradictory.

4. Also check whether the Fuzzed Query still preserves the **same target intent** as the Initial Query (i.e., still asking for the same answer node).

##### 2) Answer Leakage
Check whether the Fuzzed Query contains the exact answer string **"{answer}"** (case-sensitive).
- If it contains the exact string, leakage exists and the check fails.
- Set `leakage_ok=true` iff no exact answer string appears.

##### 3) Semantic & Type Matching
- Is the Fuzzed Query fluent and readable?
- Does it unambiguously ask for the entity/value represented by `{answer}`?
- Does the asked type match `{answer_node_type}` / `{answer_category}`?
Set `semantic_ok=true` iff fluent + unambiguous referent + type matches.

##### 4) Search Necessity Evaluation (Deep Search-ness)
Decide whether solving the Fuzzed Query reasonably requires **multi-hop web search + reasoning**.
Set `deep_search_ok=true` iff:
- It likely requires multiple searches due to obfuscation, multi-hop constraints, cross-document linkage, or intermediate facts.
Set `deep_search_ok=false` if it is likely solvable via:
- one-hop search (single query returns the answer directly), OR
- common internal knowledge (high-frequency entity/fact).
Explain why single-hop search is unlikely when `deep_search_ok=true`.

##### 5) Time Validity
If the question is time-sensitive (e.g., “current president”, changing roles/titles) and lacks a timeframe/reference point:
- set `time_validity_ok=false`
Otherwise `time_validity_ok=true`.

---

#### C) Output Requirements (JSON ONLY)
Return exactly the following JSON structure. Reasoning must come first for each dimension, then the final verdict.

All booleans in `final_verdict` mean **passes the check / OK**.
`is_legal=true` iff **all** of:
`hallucination_ok`, `leakage_ok`, `semantic_ok`, `deep_search_ok`, `time_validity_ok`
are true.

{{
  "reasoning": {{
    "hallucination": "1) List constraints extracted from fuzzed_query; 2) For each, cite which evidence source supports/contradicts it (context_text/node_details/relations/entity_replace_context); 3) Apply the default non-hallucination rule + numeric tolerance rule; 4) Conclude pass/fail for hallucination_ok.",
    "leakage": "Whether fuzzed_query contains the exact expected_answer string (case-sensitive) and therefore passes leakage_ok.",
    "semantic_and_type": "Fluency + whether the query refers to the expected answer + type match check; conclude semantic_ok.",
    "search_necessity": "Whether solving requires multi-hop search and why; conclude deep_search_ok.",
    "time_validity": "Whether time-sensitive and whether timeframe is specified; conclude time_validity_ok."
  }},
  "final_verdict": {{
    "hallucination_ok": true/false,
    "leakage_ok": true/false,
    "semantic_ok": true/false,
    "deep_search_ok": true/false,
    "time_validity_ok": true/false,
    "is_legal": true/false
  }}
}}

---

### INPUT DATA

**1) context_text**:
{context_text}

**2) entity_replace_context**:
{entity_replace_context}

**3) node_details**:
{node_details}

**4) involved_relations**:
{involved_relations}

**5) fuzzed_query**:
{fuzzed_query}

**6) expected_answer**:
{answer}

**7) answer_metadata**:
- node_type: {answer_node_type}
- category: {answer_category}

### OUTPUT
"""

class DataManager:
    """Manage cross-Phase data loading and mapping"""
    def __init__(self, data_dir: str):
        """
        Initialize data manager
        
        Args:
            data_dir: Root directory of historical Phase data (contains phase1~phase6 subdirectories)
        """
        self.data_dir = data_dir
        self.cache = {
            "phase1": {}, # source_graph_id -> graph_dict
            "phase2": {}, # source_graph_id -> {subgraph_id -> subgraph_dict}
            "phase3": {}, # source_graph_id -> {subgraph_id -> candidate_dict}
            "phase4": {}, # source_graph_id -> {subgraph_id -> context_dict}
            "phase5": {}, # source_graph_id -> {subgraph_id -> qa_dict}
            "phase6": {}, # source_graph_id -> {subgraph_id -> fuzzed_qa_dict}
        }
        # Add missing file marker cache to avoid repeated attempts to load non-existent files
        self.missing_files = set()

    def load_batch_results(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Load batch_results.jsonl file
        
        Args:
            input_file: Input file path (must be provided)
        
        Returns:
            List[Dict[str, Any]]: List of samples
        """
        if not input_file:
            raise ValueError("input_file must be provided")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Missing input file: {input_file}")
        
        results = []
        # Read and parse at once
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc="Loading batch results", unit="line"):
            line = line.strip()
            if line:
                results.append(json.loads(line))
        return results

    def _get_phase1_graph(self, source_graph_id: str) -> Optional[Dict]:
        if source_graph_id not in self.cache["phase1"]:
            path = os.path.join(self.data_dir, "phase1", f"{source_graph_id}_full_graph.json")
            # If already marked as missing, return None directly
            if path in self.missing_files:
                self.cache["phase1"][source_graph_id] = None
            elif os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    self.cache["phase1"][source_graph_id] = json.load(f)
            else:
                self.missing_files.add(path)
                self.cache["phase1"][source_graph_id] = None
        return self.cache["phase1"].get(source_graph_id)

    def _get_phase2_subgraphs(self, source_graph_id: str) -> Dict[str, Dict]:
        if source_graph_id not in self.cache["phase2"]:
            path = os.path.join(self.data_dir, "phase2", f"{source_graph_id}_subgraphs.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    save_map = {x["graph_id"]: x for x in data["subgraphs"]}
                    self.cache["phase2"][source_graph_id] = save_map
            else:
                self.cache["phase2"][source_graph_id] = {}
        return self.cache["phase2"].get(source_graph_id, {})

    def _get_phase3_candidates(self, source_graph_id: str) -> Dict[str, Dict]:
        if source_graph_id not in self.cache["phase3"]:
            path = os.path.join(self.data_dir, "phase3", f"{source_graph_id}_candidates.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    save_map = {x["graph_id"]: x for x in data["candidates"]}
                    self.cache["phase3"][source_graph_id] = save_map
            else:
                self.cache["phase3"][source_graph_id] = {}
        return self.cache["phase3"].get(source_graph_id, {})

    def _get_phase4_contexts(self, source_graph_id: str) -> Dict[str, Dict]:
        if source_graph_id not in self.cache["phase4"]:
            path = os.path.join(self.data_dir, "phase4", f"{source_graph_id}_contexts.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    save_map = {x["graph_id"]: x for x in data["contexts"]}
                    self.cache["phase4"][source_graph_id] = save_map
            else:
                self.cache["phase4"][source_graph_id] = {}
        return self.cache["phase4"].get(source_graph_id, {})

    def _get_phase5_qas(self, source_graph_id: str) -> Dict[str, Dict]:
        if source_graph_id not in self.cache["phase5"]:
            path = os.path.join(self.data_dir, "phase5", f"{source_graph_id}_qa_pairs.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    save_map = {x["metadata"]["graph_id"]: x for x in data["qa_pairs"]}
                    self.cache["phase5"][source_graph_id] = save_map
            else:
                self.cache["phase5"][source_graph_id] = {}
        return self.cache["phase5"].get(source_graph_id, {})

    def _get_phase6_fuzzed_qas(self, source_graph_id: str) -> Dict[str, Dict]:
        if source_graph_id not in self.cache["phase6"]:
            path = os.path.join(self.data_dir, "phase6", f"{source_graph_id}_fuzzed_qa.json")
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    save_map = {x["metadata"]["original_metadata"]["graph_id"]: x for x in data["fuzzed_qa_pairs"]}
                    self.cache["phase6"][source_graph_id] = save_map
            else:
                self.cache["phase6"][source_graph_id] = {}
        return self.cache["phase6"].get(source_graph_id, {})

    def get_all_historical_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and return all historical Phase data corresponding to sample.
        If any phase data fails to load, raise RuntimeError.
        """
        subgraph_id = sample.get("metadata", {}).get("original_metadata", {}).get("graph_id")
        source_graph_id = sample.get("metadata", {}).get("source_graph_id", "")

        if not source_graph_id or not subgraph_id:
            raise ValueError(f"Sample metadata missing source_graph_id or subgraph_id: {sample.get('id')}")

        # Phase 2 uses base subgraph ID (without _ans_N)
        base_subgraph_id = subgraph_id.rsplit('_ans_', 1)[0]

        # Phase 1
        p1 = self._get_phase1_graph(source_graph_id)
        if not p1:
            raise RuntimeError(f"Failed to load Phase 1 data for {source_graph_id}")

        # Phase 2
        p2_map = self._get_phase2_subgraphs(source_graph_id)
        p2 = p2_map.get(base_subgraph_id)
        if not p2:
            raise RuntimeError(f"Failed to load Phase 2 data for {base_subgraph_id} in {source_graph_id}")

        # Phase 3
        p3_map = self._get_phase3_candidates(source_graph_id)
        p3 = p3_map.get(subgraph_id)
        if not p3:
            raise RuntimeError(f"Failed to load Phase 3 data for {subgraph_id} in {source_graph_id}")

        # Phase 4
        p4_map = self._get_phase4_contexts(source_graph_id)
        p4 = p4_map.get(subgraph_id)
        if not p4:
            raise RuntimeError(f"Failed to load Phase 4 data for {subgraph_id} in {source_graph_id}")

        # Phase 5
        p5_map = self._get_phase5_qas(source_graph_id)
        p5 = p5_map.get(subgraph_id)
        if not p5:
            raise RuntimeError(f"Failed to load Phase 5 data for {subgraph_id} in {source_graph_id}")

        # Phase 6
        p6_map = self._get_phase6_fuzzed_qas(source_graph_id)
        p6 = p6_map.get(subgraph_id)
        if not p6:
            raise RuntimeError(f"Failed to load Phase 6 data for {subgraph_id} in {source_graph_id}")

        return {
            "phase1": p1,
            "phase2": p2,
            "phase3": p3,
            "phase4": p4,
            "phase5": p5,
            "phase6": p6
        }

    def get_full_context_data(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate data from each phase to build complete validation context"""
        subgraph_id = sample.get("metadata", {}).get("original_metadata", {}).get("graph_id")
        source_graph_id = sample.get("metadata", {}).get("source_graph_id", "")

        if not source_graph_id or not subgraph_id:
            return {}
        
        # load subgraph
        subgraph_id_wo_ans = subgraph_id.rsplit('_ans_', 1)[0]
        subgraphs = self._get_phase2_subgraphs(source_graph_id)
        subgraph = subgraphs.get(subgraph_id_wo_ans, {})

        # 1. Get Phase 5 original QA
        phase5_qas = self._get_phase5_qas(source_graph_id)
        qa_pair = phase5_qas.get(subgraph_id, {})
        
        metadata = qa_pair.get('metadata', {})
        p1_fuzz = metadata.get('phase_1_fuzzification', {})
        entity_replace_context = p1_fuzz.get('fuzzified_context_text', "N/A")
        
        p2_puzzle = metadata.get('phase_2_puzzle_design', {})
        initial_query = p2_puzzle.get('final_complex_query', "N/A")
        
        # 2. Get Phase 4 context
        phase4_contexts = self._get_phase4_contexts(source_graph_id)
        context_obj = phase4_contexts.get(subgraph_id, {})
        
        # 3. Get detailed Node and Relation info
        involved_nodes_ids = context_obj.get("involved_nodes", [])
        node_details = []
        involved_relations = context_obj.get("involved_relations", [])
        
        full_graph = self._get_phase1_graph(source_graph_id)
        if full_graph:
            kg = full_graph.get("knowledge_graph", {})
            all_nodes = kg.get("nodes", {})
            all_rels = kg.get("relations", [])
            node_set = set(involved_nodes_ids)
            
            # Extract Node details
            for nid in involved_nodes_ids:
                n_info = all_nodes.get(nid, {})
                node_details.append({
                    "id": nid,
                    "category": n_info.get("category"),
                    "description": n_info.get("description"),
                    # "grounding": n_info.get("grounding_fragment")
                })
            
            # If Phase 4 lacks relation info, supplement from Phase 1
            if not involved_relations:
                involved_relations = [
                    f"{r['source_id']} --({r['relation']})--> {r['target_id']}"
                    for r in all_rels 
                    if r['source_id'] in node_set and r['target_id'] in node_set
                ]

        return {
            "node_details": node_details,
            "involved_relations": involved_relations,
            "context_text": context_obj.get("context_text", ""),
            "entity_replace_context": entity_replace_context,
            "initial_query": initial_query,
            "fuzzed_query": sample.get("fuzzed_query", ""),
            "answer": sample.get("answer", ""),
            "answer_node_type": context_obj.get("answer_node_type", "unknown"),
            "answer_category": context_obj.get("answer_category", "unknown")
        }


def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract and parse JSON object from LLM response.
    Supports handling Markdown tags, leading/trailing impurity text, etc.
    """
    if not response:
        return {}
    
    # 1. Try direct parsing
    content = response.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. Try extracting from Markdown code block (```json ... ``` or ``` ... ```)
    import re
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(code_block_pattern, content)
    if match:
        code_content = match.group(1).strip()
        try:
            return json.loads(code_content)
        except json.JSONDecodeError:
            content = code_content # If extracted code block still fails to parse, continue with subsequent cleanup logic

    # 3. Find content between first '{' and last '}' (strongest cleanup)
    start_idx = content.find('{')
    end_idx = content.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        json_str = content[start_idx:end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Final attempt: handle some common LLM output errors, like trailing commas
            try:
                # Remove trailing comma after last key-value pair in JSON object
                json_str_fixed = re.sub(r',\s*([\]}])', r'\1', json_str)
                return json.loads(json_str_fixed)
            except:
                pass

    return {}

# ==========================================
# Encapsulated Interfaces
# ==========================================

def check_data_integrity(data_mgr: DataManager, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interface 1: Data integrity check
    Verify if all historical phase data (Phase 1-6) corresponding to this sample exists and can be loaded correctly.
    """
    try:
        data_mgr.get_all_historical_data(sample)
        return {"id": sample.get("id"), "status": "success", "message": "Integrity check passed."}
    except Exception as e:
        return {
            "id": sample.get("id", "unknown"),
            "status": "fail",
            "error_type": "DataIntegrityError",
            "message": str(e)
        }

async def check_data_legality(client: AsyncOpenAI, model: str, data_mgr: DataManager, sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interface 2: Data legality check
    Use LLM to verify sample's logical legality (Hallucination, Leakage, Search Necessity, etc.).
    """
    data = data_mgr.get_full_context_data(sample)
    if not data:
        return {
            "id": sample.get("id"),
            "is_legal": False,
            "reason": "Failed to aggregate context data for legality check.",
            "error_type": "ContextAggregationError"
        }
    return await validate_sample(client, model, data, sample.get("id", "unknown"))


async def validate_sample(client: AsyncOpenAI, model: str, data: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
    if not data:
        return {"id": sample_id, "is_legal": False, "reason": "Data loading failed", "error_type": "InternalError", "source_data": data}

    prompt = WEBQA_VALIDATOR_PROMPT.format(
        context_text=data["context_text"],
        entity_replace_context=data["entity_replace_context"],
        node_details=json.dumps(data["node_details"], ensure_ascii=False, indent=2),
        involved_relations=json.dumps(data["involved_relations"], ensure_ascii=False, indent=2),
        fuzzed_query=data["fuzzed_query"],
        answer=data["answer"],
        answer_node_type=data["answer_node_type"],
        answer_category=data["answer_category"]
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            timeout=400
        )
        raw_result = response.choices[0].message.content
        res_json = extract_json_from_response(raw_result)
        
        if not res_json:
            return {
                "id": sample_id,
                "is_legal": False,
                "reason": f"Failed to parse JSON from LLM response: {raw_result[:200]}...",
                "error_type": "JSONParseError",
                "source_data": data
            }

        res_json["id"] = sample_id
        res_json["source_data"] = data
        return res_json
    except Exception as e:
        return {
            "id": sample_id,
            "is_legal": False,
            "reason": f"LLM Error: {str(e)}",
            "error_type": "InternalError",
            "source_data": data
        }

class DataValidator:
    """Data validator, encapsulates validation logic"""
    def __init__(self, config_path: str):
        self.config = FileUtils.load_config(config_path)
        self.data_mgr = None  # Will be initialized in worker process
        self._init_llm_endpoint()
    
    def _init_llm_endpoint(self):
        """Initialize LLM endpoint, supports random selection from endpoint_list"""
        validator_config = self.config.get("validator", {})
        endpoint_list = validator_config.get("endpoint_list", [])
        
        if endpoint_list and len(endpoint_list) > 0:
            self.llm_endpoint = random.choice(endpoint_list)
            logger.info(f"Selected endpoint: {self.llm_endpoint}")
        else:
            self.llm_endpoint = validator_config.get("endpoint", "")
            logger.info(f"Using single endpoint: {self.llm_endpoint}")
    
    def get_endpoint(self):
        """Randomly select one from endpoint_list on each call"""
        validator_config = self.config.get("validator", {})
        endpoint_list = validator_config.get("endpoint_list", [])
        
        if endpoint_list and len(endpoint_list) > 0:
            return random.choice(endpoint_list)
        else:
            return validator_config.get("endpoint", "")


def _worker_process_batch(worker_args):
    """
    Worker function for subprocess: process a batch of data assigned to this process
    Use asyncio for async processing within the process
    """
    try:
        batch_data, config_path, data_dir, threads_per_process, queue_multiplier, worker_id = worker_args
        
        # Re-configure logger in subprocess
        worker_logger = logging.getLogger(f"worker_{worker_id}")
        worker_logger.setLevel(logging.INFO)
        
        # Add console handler (if not already added)
        if not worker_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                f'[Worker {worker_id}] %(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            worker_logger.addHandler(handler)
        
        worker_logger.info(f"Worker started, processing {len(batch_data)} samples")
        worker_logger.info(f"Config path: {config_path}")
        worker_logger.info(f"Data dir: {data_dir}")
        
        # 加载配置
        config = FileUtils.load_config(config_path)
        worker_logger.info("Config loaded successfully")
    except Exception as e:
        print(f"[Worker {worker_id}] FATAL ERROR during initialization: {e}", flush=True)
        traceback.print_exc()
        return []
    
    async def process_batch_async():
        """Asynchronously process a batch of data within subprocess"""
        try:
            semaphore = asyncio.Semaphore(threads_per_process)
            max_active_tasks = threads_per_process * queue_multiplier
            
            worker_logger.info(f"Max active tasks = {max_active_tasks}")
            
            # Initialize data manager and validator
            worker_logger.info(f"Initializing DataManager with data_dir: {data_dir}")
            data_mgr = DataManager(data_dir)
            worker_logger.info("DataManager initialized successfully")
        except Exception as e:
            worker_logger.error(f"FATAL ERROR during async initialization: {e}")
            worker_logger.error(traceback.format_exc())
            raise
        
        validator_config = config.get("validator", {})
        api_key = validator_config.get("api_key", "abc")
        model_name = validator_config.get("model_name", "deepseek-v3.2")
        timeout = validator_config.get("timeout", 600)
        
        # Prepare endpoint list and round-robin counter (ensure load balancing)
        endpoint_list = validator_config.get("endpoint_list", [])
        if not endpoint_list:
            endpoint_list = [validator_config.get("endpoint", "")]
        
        worker_logger.info(f"Loaded {len(endpoint_list)} endpoints for round-robin distribution")
        
        # Use round-robin to assign endpoint (thread-safe)
        # Each worker starts from different offset, further balancing load
        start_offset = worker_id % len(endpoint_list)
        rotated_endpoints = endpoint_list[start_offset:] + endpoint_list[:start_offset]
        endpoint_cycle = itertools.cycle(rotated_endpoints)
        endpoint_lock = asyncio.Lock()
        
        worker_logger.info(f"Worker starting from endpoint index {start_offset}: {rotated_endpoints[0]}")
        
        async def process_single_sample(sample):
            """Process single sample"""
            MAX_RETRIES = 4
            sample_id = sample.get("id", "unknown")
            
            for attempt in range(MAX_RETRIES):
                try:
                    async with semaphore:
                        async with asyncio.timeout(timeout):
                            # Use round-robin to select endpoint (load balancing)
                            async with endpoint_lock:
                                endpoint = next(endpoint_cycle)
                            
                            worker_logger.debug(f"Sample {sample_id}: Using endpoint {endpoint}")
                            client = AsyncOpenAI(api_key=api_key, base_url=endpoint)
                            
                            # 1. Integrity check
                            integrity = check_data_integrity(data_mgr, sample)
                            if integrity["status"] == "fail":
                                return {
                                    "id": sample_id,
                                    "sample": sample,
                                    "validation_result": integrity,
                                    "legal": False
                                }
                            
                            # 2. Legality check
                            legality = await check_data_legality(client, model_name, data_mgr, sample)
                            
                            # Determine if passed
                            is_legal = False
                            if "final_verdict" in legality:
                                is_legal = legality["final_verdict"].get("is_legal", False)
                            else:
                                is_legal = legality.get("is_legal", False)
                            
                            return {
                                "id": sample_id,
                                "sample": sample,
                                "validation_result": legality,
                                "legal": is_legal
                            }
                
                except asyncio.TimeoutError:
                    worker_logger.error(f"Worker {worker_id}: Sample {sample_id} timed out (attempt {attempt + 1}/{MAX_RETRIES})")
                    if attempt == MAX_RETRIES - 1:
                        return {
                            "id": sample_id,
                            "sample": sample,
                            "validation_result": {
                                "error_type": "TimeoutError",
                                "message": f"Validation timed out after {timeout}s"
                            },
                            "legal": False
                        }
                    await asyncio.sleep(5)
                
                except Exception as e:
                    worker_logger.error(f"Worker {worker_id}: Failed to process sample {sample_id} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                    worker_logger.error(traceback.format_exc())
                    if attempt == MAX_RETRIES - 1:
                        return {
                            "id": sample_id,
                            "sample": sample,
                            "validation_result": {
                                "error_type": type(e).__name__,
                                "message": str(e)
                            },
                            "legal": False
                        }
                    await asyncio.sleep(5)
            
            return None
        
        # Use limited queue to control active task count
        pending_data = list(batch_data)
        active_tasks = {}
        results = []
        completed_count = 0
        
        # Initialize queue
        while len(active_tasks) < max_active_tasks and pending_data:
            sample = pending_data.pop(0)
            task = asyncio.create_task(process_single_sample(sample))
            active_tasks[task] = {"id": sample.get("id", "unknown")}
        
        worker_logger.info(f"Worker {worker_id}: Initial queue filled with {len(active_tasks)} tasks")
        
        # 持续处理直到所有任务完成
        while active_tasks:
            done, _ = await asyncio.wait(
                active_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                task_info = active_tasks[task]
                completed_count += 1
                
                try:
                    result = await task
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    worker_logger.error(f"Worker {worker_id}: Task for sample {task_info['id']} failed: {e}")
                
                del active_tasks[task]
            
            # Add new tasks
            while len(active_tasks) < max_active_tasks and pending_data:
                sample = pending_data.pop(0)
                task = asyncio.create_task(process_single_sample(sample))
                active_tasks[task] = {"id": sample.get("id", "unknown")}
            
            if completed_count > 0 and completed_count % 10 == 0:
                worker_logger.info(f"Worker {worker_id}: Progress {completed_count}/{len(batch_data)}")
        
        worker_logger.info(f"Worker {worker_id}: Completed {len(results)}/{len(batch_data)} samples")
        return results
    
    # Create new event loop in subprocess and run
    try:
        worker_logger.info("Creating event loop")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        worker_logger.info("Starting async batch processing")
        results = loop.run_until_complete(process_batch_async())
        loop.close()
        worker_logger.info(f"Worker finished successfully with {len(results)} results")
        return results
    except Exception as e:
        error_msg = f"Worker failed with exception: {type(e).__name__}: {e}"
        worker_logger.error(error_msg)
        worker_logger.error(traceback.format_exc())
        # 同时输出到标准错误，确保能看到
        print(f"[Worker {worker_id}] ERROR: {error_msg}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return []


def main_multiprocess(args):
    """Multi-process + async mode main function"""
    import time
    
    config = FileUtils.load_config(args.config_path)
    exp_config = config.get("exps", {})
    validator_config = config.get("validator", {})
    
    # 获取配置参数
    input_file = exp_config.get("input_file", args.input_file)
    data_dir = exp_config.get("data_dir", "")
    output_dir = exp_config.get("output_dir", "")
    num_processes = exp_config.get("num_processes", mp.cpu_count())
    threads_per_process = exp_config.get("threads_per_process", 8)
    queue_multiplier = exp_config.get("queue_multiplier", 2)
    
    # Parameter validation: must be provided
    if not input_file:
        raise ValueError("input_file must be specified in config file (exps.input_file)")
    if not data_dir:
        raise ValueError("data_dir must be specified in config file (exps.data_dir)")
    if not output_dir:
        raise ValueError("output_dir must be specified in config file (exps.output_dir)")
    
    # Verify path existence
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    start_time = time.time()
    
    logger.info("="*80)
    logger.info("Starting DeepQA Data Validator")
    logger.info("="*80)
    logger.info(f"Input file:     {input_file}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output dir:     {output_dir}")
    logger.info(f"Processes:      {num_processes}")
    logger.info(f"Threads/proc:   {threads_per_process}")
    logger.info("="*80)
    
    # Save config to output directory
    os.makedirs(output_dir, exist_ok=True)
    FileUtils.save_json(config, f"{output_dir}/config.json")
    logger.info(f"Config saved to {output_dir}/config.json")
    
    # Load input data
    logger.info(f"Loading samples from {input_file}...")
    data_mgr = DataManager(data_dir)
    samples = data_mgr.load_batch_results(input_file)
    logger.info(f"✓ Loaded {len(samples)} samples successfully")
    
    # Create problems directory
    problems_dir = f"{output_dir}/problems"
    os.makedirs(problems_dir, exist_ok=True)
    
    # Check processed files
    logger.info("Checking existing files...")
    existing_files = set()
    if os.path.exists(problems_dir):
        try:
            all_files = os.listdir(problems_dir)
            for filename in all_files:
                if filename.startswith('problem_') and filename.endswith('.json'):
                    try:
                        # Extract 'Q123' from 'problem_Q123.json'
                        idx_str = filename.replace('problem_', '').replace('.json', '')
                        existing_files.add(idx_str)
                    except ValueError:
                        logger.warning(f"Skipping invalid filename: {filename}")
            logger.info(f"Found {len(existing_files)} existing files")
        except Exception as e:
            logger.warning(f"Failed to list directory {problems_dir}: {e}")
    
    # Filter processed samples
    filtered_samples = []
    skipped_count = 0
    
    for sample in samples:
        sample_id = str(sample.get("id", "unknown"))
        
        if sample_id in existing_files and not args.override:
            skipped_count += 1
            continue
        
        filtered_samples.append(sample)
    
    logger.info(f"Skipped {skipped_count} existing files")
    logger.info(f"Total {len(filtered_samples)} tasks to process")
    
    if len(filtered_samples) == 0:
        logger.info("No tasks to process. Exiting.")
        return
    
    # Distribute batches
    batch_size = (len(filtered_samples) + num_processes - 1) // num_processes
    batches = []
    for i in range(num_processes):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(filtered_samples))
        if start_idx < len(filtered_samples):
            batches.append(filtered_samples[start_idx:end_idx])
    
    logger.info("="*80)
    logger.info(f"Starting multiprocess validation:")
    logger.info(f"  Total samples:        {len(filtered_samples)}")
    logger.info(f"  Number of processes:  {len(batches)}")
    logger.info(f"  Threads per process:  {threads_per_process}")
    logger.info(f"  Queue multiplier:     {queue_multiplier}")
    logger.info(f"  Batch sizes:          {[len(b) for b in batches]}")
    logger.info("="*80)
    
    # Prepare worker arguments
    logger.info("Preparing worker arguments...")
    worker_args_list = [
        (batch, args.config_path, data_dir, threads_per_process, queue_multiplier, worker_id)
        for worker_id, batch in enumerate(batches)
    ]
    logger.info(f"✓ Prepared {len(worker_args_list)} worker tasks")
    
    # Execute multi-process processing
    all_results = []
    try:
        logger.info(f"Creating process pool with {len(batches)} workers...")
        with Pool(processes=len(batches)) as pool:
            logger.info("✓ Process pool created successfully")
            logger.info("Submitting batches to process pool...")
            
            with tqdm(
                total=len(filtered_samples),
                desc="Validating samples",
                unit="sample",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            ) as pbar:
                completed_batches = 0
                logger.info("Waiting for worker results...")
                for batch_result in pool.imap_unordered(_worker_process_batch, worker_args_list, chunksize=1):
                    logger.debug(f"Received batch result with {len(batch_result)} items")
                    # 保存每个样本到独立文件
                    for result in batch_result:
                        sample_id = result["id"]
                        save_path = f"{problems_dir}/problem_{sample_id}.json"
                        FileUtils.save_json(result, save_path)
                        all_results.append(result)
                    
                    completed_batches += 1
                    pbar.update(len(batch_result))
                    pbar.set_postfix({
                        'batches': f'{completed_batches}/{len(batches)}',
                        'completed': len(all_results)
                    })
                    pbar.refresh()
            
            logger.info(f"All workers finished. Total results: {len(all_results)}/{len(filtered_samples)}")
    
    except Exception as e:
        logger.error(f"Error in multiprocess execution: {e}")
        logger.error(traceback.format_exc())
        raise
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    if len(all_results) > 0:
        legal_count = sum(1 for r in all_results if r.get("legal", False))
        illegal_count = len(all_results) - legal_count
        
        # Calculate pass rates for each dimension
        dimension_stats = {
            "hallucination_ok": 0,
            "leakage_ok": 0,
            "semantic_ok": 0,
            "deep_search_ok": 0,
            "time_validity_ok": 0
        }
        
        for result in all_results:
            validation_result = result.get("validation_result", {})
            final_verdict = validation_result.get("final_verdict", {})
            
            for key in dimension_stats.keys():
                if final_verdict.get(key, False):
                    dimension_stats[key] += 1
        
        # Calculate ratios
        total_validated = len(all_results)
        dimension_rates = {
            key: count / total_validated if total_validated > 0 else 0
            for key, count in dimension_stats.items()
        }
        
        overall_stats = {
            "total_time": total_time,
            "total_samples": len(filtered_samples) + skipped_count,
            "processed_samples": len(all_results),
            "skipped_samples": skipped_count,
            "legal_count": legal_count,
            "illegal_count": illegal_count,
            "legal_rate": legal_count / total_validated if total_validated > 0 else 0,
            "illegal_rate": illegal_count / total_validated if total_validated > 0 else 0,
            "dimension_stats": dimension_stats,
            "dimension_rates": dimension_rates,
            "samples_per_second": len(all_results) / total_time if total_time > 0 else 0,
            "num_processes": num_processes,
            "threads_per_process": threads_per_process
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Validation completed!")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Processed: {len(all_results)}/{len(filtered_samples) + skipped_count}")
        logger.info(f"  Legal samples: {legal_count} ({overall_stats['legal_rate']:.2%})")
        logger.info(f"  Illegal samples: {illegal_count} ({overall_stats['illegal_rate']:.2%})")
        logger.info(f"  Samples per second: {overall_stats['samples_per_second']:.2f}")
        logger.info(f"\n  Dimension pass rates:")
        for dim, rate in dimension_rates.items():
            logger.info(f"    {dim}: {rate:.2%}")
        logger.info(f"{'='*60}")
        
        # Save results
        results_data = {
            "statistics": overall_stats,
            "results": all_results
        }
        
        FileUtils.save_json(results_data, f"{output_dir}/results.json")
        FileUtils.save_json({
            "legal_count": legal_count,
            "illegal_count": illegal_count,
            "legal_rate": overall_stats["legal_rate"],
            "illegal_rate": overall_stats["illegal_rate"],
            "dimension_rates": dimension_rates
        }, f"{output_dir}/accuracy.json")
        
        logger.info(f"Results saved to {output_dir}/results.json and accuracy.json")
    else:
        logger.warning("No results to save")


def parse_args():
    parser = argparse.ArgumentParser(description="DeepQA Phase 6 Data Validator")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--input_file", type=str, default="", help="Input JSONL file (overrides config)")
    parser.add_argument("--override", action="store_true", default=False, help="Override existing results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main_multiprocess(args)
