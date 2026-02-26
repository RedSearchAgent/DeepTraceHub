import os
import sys
import json
import time
import wandb
import asyncio
import argparse
import traceback
import numpy as np
import multiprocessing as mp
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2])
    sys.path.append(base_path)

from tools.search import (
    EmptyWebSearchTool,
    SerperDevSearchTool,
    LocalWebSearchTool,
)
from tools.search import (
    SerperDevKeyManager,
)
from tools.summarizer import (
    LLMSummarizer,
    BriefLLMSummarizer,
)

from tools.search import (
    JinaWebCrawlTool,
    SerperDevWebCrawlTool,
    LocalWebCrawlTool,
)

from utils.file_utils import FileUtils
from utils.logger import get_logger


# Import all Agent classes from src.agents module
from src.agents import (
    AgentBase,
    MODEL_ARGS,
    log_exception_details,
    DirectQAVerifier,
    ReACTAgent,
    DeepSeekV32ThinkingWithToolAgent,
)


TOOL_CLASS_SEARCH = [
    EmptyWebSearchTool,
    SerperDevSearchTool,
    LocalWebSearchTool,
]
TOOL_CLASS_CRAWL = [
    JinaWebCrawlTool,
    SerperDevWebCrawlTool,
    LocalWebCrawlTool,
]
TOOL_CLASS_SUMMARIZER = [
    LLMSummarizer,
    BriefLLMSummarizer,
]
TOOL_MAP_SEARCH = {item.name: item for item in TOOL_CLASS_SEARCH}
TOOL_MAP_CRAWL = {item.name: item for item in TOOL_CLASS_CRAWL}
TOOL_MAP_SUMMARIZER = {item.name: item for item in TOOL_CLASS_SUMMARIZER}
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="config/config.yaml")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--override", action="store_true", default=False)
    parser.add_argument("--multiprocess", action="store_true", default=True, 
                        help="Use multiprocess + asyncio mode for better parallelism")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume incomplete tasks (files with 'resume': true)")
    return parser.parse_args()


def _filter_completed_tasks(flatten_datas: list, output_dir: str, resume_mode: bool = False) -> tuple:
    """
    Efficiently filter completed tasks to avoid checking one by one inside process_query.
    
    Optimization strategies:
    1. Use os.listdir to batch get directory file lists (much faster than checking os.path.exists one by one)
    2. Use multi-threading to read JSON files in parallel in resume mode
    
    Args:
        flatten_datas: Flattened task list
        output_dir: Output directory
        resume_mode: Whether in resume mode (need to check file content to determine completion)
    
    Returns:
        (filtered_datas, skipped_count, resume_count)
    """
    # Group by dataset_id to get existing files
    # First collect all involved dataset_ids
    dataset_ids = set(d["dataset_id"] for d in flatten_datas)
    
    # Batch get file lists in each directory
    existing_files_by_dataset = {}
    for dataset_id in dataset_ids:
        # Handle empty dataset_id case (single file mode)
        dataset_dir = f"{output_dir}/{dataset_id}" if dataset_id else output_dir
        if os.path.exists(dataset_dir):
            try:
                # Get all files in directory at once, much faster than checking os.path.exists one by one
                files = set(f for f in os.listdir(dataset_dir) 
                           if f.endswith('.json') and f.startswith('problem_'))
                existing_files_by_dataset[dataset_id] = files
            except Exception as e:
                logger.warning(f"Failed to list directory {dataset_dir}: {e}")
                existing_files_by_dataset[dataset_id] = set()
        else:
            existing_files_by_dataset[dataset_id] = set()
    
    # Build filename for each task
    def get_filename(data):
        idx = data["idx"]
        sample_idx = data.get("sample_idx", 0)
        if sample_idx > 0:
            return f"problem_{idx}_sample_{sample_idx}.json"
        else:
            return f"problem_{idx}.json"
    
    if not resume_mode:
        # Non-resume mode: skip all existing files (no need to read content)
        filtered_datas = []
        skipped_count = 0
        for data in flatten_datas:
            dataset_id = data["dataset_id"]
            filename = get_filename(data)
            if filename in existing_files_by_dataset.get(dataset_id, set()):
                skipped_count += 1
            else:
                filtered_datas.append(data)
        
        logger.info(f"Filtered {skipped_count} existing files (non-resume mode)")
        return filtered_datas, skipped_count, 0
    
    else:
        # Resume mode: need to check file content to determine if need to continue processing
        # Find all files that need content checking
        files_to_check = []
        data_by_file = {}
        for data in flatten_datas:
            dataset_id = data["dataset_id"]
            filename = get_filename(data)
            if filename in existing_files_by_dataset.get(dataset_id, set()):
                # Handle empty dataset_id case (single file mode)
                file_path = f"{output_dir}/{dataset_id}/{filename}" if dataset_id else f"{output_dir}/{filename}"
                files_to_check.append(file_path)
                data_by_file[file_path] = data
        
        if not files_to_check:
            # 没有需要检查的文件
            return flatten_datas, 0, 0
        
        # Use multi-threading to read JSON files in parallel
        def check_file_status(file_path):
            """Check single file status, return (file_path, is_completed, needs_resume)"""
            try:
                with open(file_path, 'r') as f:
                    saved_data = json.load(f)
                is_resume = saved_data.get("resume", False)
                return (file_path, not is_resume, is_resume)
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}, will reprocess")
                return (file_path, False, False)
        
        # Read files in parallel (using reasonable number of threads)
        max_workers = min(64, len(files_to_check))
        completed_files = set()
        resume_count = 0
        
        logger.info(f"Checking {len(files_to_check)} existing files with {max_workers} threads...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(check_file_status, files_to_check))
        
        for file_path, is_completed, needs_resume in results:
            if is_completed:
                completed_files.add(file_path)
            if needs_resume:
                resume_count += 1
        
        # Filter task list
        filtered_datas = []
        skipped_count = 0
        for data in flatten_datas:
            dataset_id = data["dataset_id"]
            filename = get_filename(data)
            # Handle empty dataset_id case (single file mode)
            file_path = f"{output_dir}/{dataset_id}/{filename}" if dataset_id else f"{output_dir}/{filename}"
            
            if file_path in completed_files:
                skipped_count += 1
            else:
                filtered_datas.append(data)
        
        logger.info(f"Filtered {skipped_count} completed files, {resume_count} files need resume")
        return filtered_datas, skipped_count, resume_count


def run_async_query(agent, idx, query, ground_truth, override=False):
    """Wrap async method for multi-threading use"""
    return asyncio.run(agent.process_query(idx, query, ground_truth=ground_truth, override=override))


def process_batch_worker(batch_data, process_id, task_config):
    """Multi-process worker function, process a batch of data"""
    logger.info(f"Process {process_id} started, processing {len(batch_data)} queries")
    config = FileUtils.load_config(task_config["config_path"])
    key_manager = SerperDevKeyManager(config["tools"]["serper_key_usage_file"])
    # Create new agent instance in each process
    logger.info(f"Creating agent instance: {config['agent']['cls']}")
    agent = eval(config["agent"]["cls"])(
        config_path=task_config["config_path"],
        key_manager=key_manager
    )
    
    # Use multi-threading to process current batch of data
    max_workers = min(task_config.get("threads_per_process", 4), len(batch_data))
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_async_query, agent, idx=data["idx"], query=data["problem"], ground_truth=data["answer"]) 
                  for data in batch_data]
        results = [future.result() for future in futures]
    
    # Filter out None results
    results = [result for result in results if result is not None]
    
    logger.info(f"Process {process_id} completed, processed {len(results)} queries successfully")
    return results

def main_debug(config_path, override=False):

    config = FileUtils.load_config(config_path)
    if not os.path.exists(config["exps"]["output_dir"]):
        os.makedirs(config["exps"]["output_dir"], exist_ok=True)
    datas = FileUtils.load_jsonl(config["exps"]["input_file"])
    key_manager = SerperDevKeyManager(config["tools"]["serper_key_usage_file"])

    logger.info(f"Creating agent instance: {config['agent']['cls']}")
    agent = eval(config["agent"]["cls"])(
        config_path=config_path,
        key_manager=key_manager
    )
    #async def process_query(self, idx: int, user_query: str, ground_truth: str = None):
    for data in datas:
        result = asyncio.run(agent.process_query(idx=data["idx"], user_query=data["problem"], ground_truth=data["answer"], override=override))
        print(result)

async def main_asyncio_multiple_files(args):
    """Async version of main function, using asyncio.Semaphore to maintain a pool with capacity of num_processes, processing multiple file data simultaneously"""
    config = FileUtils.load_config(args.config_path)
    exp_config = config["exps"]
    input_file_list = exp_config["input_file_list"]
    if not isinstance(input_file_list, list):
        input_file_list = [input_file_list]
    
    # Get config parameters
    num_processes = exp_config.get("num_processes", mp.cpu_count())
    max_workers = exp_config.get("max_workers", 512)
    
    # Calculate actual concurrency limit
    concurrency_limit = min(num_processes, max_workers)
    start_time = time.time()
    FileUtils.save_json(config, f"{config['exps']['output_dir']}/config.json")
    
    # Initialize wandb if enabled
    wandb_enabled = config.get("wandb", {}).get("enabled", False)
    if wandb_enabled:
        os.environ["WANDB_API_KEY"] = config.get("wandb", {}).get("wandb_api_key", "")
        wandb_config = config.get("wandb", {})
        wandb.init(
            project=wandb_config.get("project", "deeptracehub"),
            name=wandb_config.get("run_name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            config={
                "agent_config": config.get("agent", {}),
                "llm_config": config.get("llm", {}),
                "tools_config": config.get("tools", {}),
                "exps_config": config.get("exps", {})
            },
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", "")
        )


    # Read data from multiple files
    datas = {}
    for input_file in input_file_list:
        dataset_id = os.path.basename(input_file).split(".")[0]
        datas[dataset_id] = FileUtils.load_jsonl(input_file)
        logger.info(f"Loaded {len(datas[dataset_id])} queries from {input_file}")

    for dataset_id in datas.keys():
        if not os.path.exists(f"{config['exps']['output_dir']}/{dataset_id}"):
            os.makedirs(f"{config['exps']['output_dir']}/{dataset_id}", exist_ok=True)
            logger.info("Create directory: ", f"{config['exps']['output_dir']}/{dataset_id}")

    # Flatten data and expand samples according to sampling config
    flatten_datas = []
    sampling_enabled = config["agent"].get("sampling", {}).get("enabled", False)
    n_samples = config["agent"]["sampling"]["n_samples"] if sampling_enabled else 1
    
    for dataset_id, dataset_datas in datas.items():
        for data in dataset_datas:
            if sampling_enabled:
                # Create n_samples independent tasks for each query
                for sample_idx in range(n_samples):
                    sample_data = {
                        "dataset_id": dataset_id,
                        "idx": data["idx"],
                        "problem": data["problem"],
                        "answer": data["answer"],
                        "sample_idx": sample_idx,  # Mark which sample this is
                        "n_samples": n_samples      # Total number of samples
                    }
                    flatten_datas.append(sample_data)
            else:
                # Single sample mode (original logic)
                data["dataset_id"] = dataset_id
                data["sample_idx"] = 0
                data["n_samples"] = 1
                flatten_datas.append(data)
    
    if sampling_enabled:
        logger.info(f"Flattened {len(flatten_datas)} tasks (samples) from {len(datas)} datasets")
        logger.info(f"Each query has {n_samples} samples, total queries: {sum(len(v) for v in datas.values())}")
    else:
        logger.info(f"Flattened {len(flatten_datas)} queries from {len(datas)} datasets")

    # Pre-filter completed tasks (avoid checking one by one inside process_query)
    output_dir = config['exps']['output_dir']
    original_count = len(flatten_datas)
    flatten_datas, skipped_count, resume_count = _filter_completed_tasks(
        flatten_datas, output_dir, resume_mode=args.resume
    )
    logger.info(f"Pre-filtered: {original_count} -> {len(flatten_datas)} tasks ({skipped_count} skipped, {resume_count} to resume)")

    # Create shared key_manager (avoid duplicate creation)
    key_manager = SerperDevKeyManager(config["tools"]["serper_key_usage_file"])
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def process_single_query(data):
        """Process single sample (not all samples of the entire query)"""
        TIMEOUT = 36000 # 10 hours
        MAX_RETRIES = 4
        
        for attempt in range(MAX_RETRIES):
            try:
                # First acquire semaphore (waiting time not counted in timeout)
                async with semaphore:  # Each sample occupies one concurrency slot
                    # Start timing after acquiring semaphore (only count actual processing time)
                    async with asyncio.timeout(TIMEOUT):
                        # Create independent agent instance for each task, but share key_manager
                        agent = eval(config["agent"]["cls"])(
                            config_path=args.config_path,
                            key_manager=key_manager
                        )
                        
                        # Pass sample_idx parameter
                        result = await agent.process_query(
                            idx=data["idx"], 
                            user_query=data["problem"], 
                            ground_truth=data["answer"],
                            dataset_id=data["dataset_id"],
                            sample_idx=data.get("sample_idx", 0),  # Pass sample index
                            override=args.override
                        )
                        
                        # Record sample info in result
                        if result is not None:
                            result["sample_idx"] = data.get("sample_idx", 0)
                            result["n_samples"] = data.get("n_samples", 1)
                        
                        return result
            except asyncio.TimeoutError as e:
                logger.error(f"Query {data['idx']} sample {data.get('sample_idx', 0)} timed out after {TIMEOUT} seconds (attempt {attempt + 1}/{MAX_RETRIES})")
                logger.error(traceback.format_exc())
                if attempt == MAX_RETRIES - 1:
                    return None
                await asyncio.sleep(5)  # Wait 5 seconds before retry
            except Exception as e:
                logger.error(f"Failed to process query {data['idx']} sample {data.get('sample_idx', 0)} (attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}: {e}")
                logger.error(traceback.format_exc())
                if attempt == MAX_RETRIES - 1:
                    return None
                await asyncio.sleep(5)  # Wait 5 seconds before retry
        
        return None

    logger.info(f"Starting asyncio batch processing:")
    logger.info(f"  Total queries: {len(flatten_datas)}")
    logger.info(f"  Concurrency limit: {concurrency_limit}")
    logger.info(f"  Max workers: {max_workers}")
    
    # Use asyncio.gather to execute all tasks concurrently
    all_results = []
    
    try:
        logger.info("Submitting tasks to asyncio...")
        # 创建所有任务
        tasks = [process_single_query(data) for data in flatten_datas]
        
        # 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        for result in results:
            if result is not None and not isinstance(result, Exception):
                all_results.append(result)

                
                
    except Exception as e:
        logger.error(f"Error in asyncio execution: {e}")
        raise

    end_time = time.time()
    total_time = end_time - start_time

    # Group results by dataset_id and query_idx
    results_by_dataset_and_query = defaultdict(lambda: defaultdict(list))
    for result in all_results:
        dataset_id = result["dataset_id"]
        query_idx = result["idx"]  # Now all results have idx field
        results_by_dataset_and_query[dataset_id][query_idx].append(result)

    # Calculate outermost statistics
    processing_times = [result.get("processing_time_seconds", 0) for result in all_results]
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    max_time = max(processing_times) if processing_times else 0
    min_time = min(processing_times) if processing_times else 0
    
    total_queries = sum(len(v) for v in datas.values())
    logger.info(f"Asyncio batch processing completed:")
    logger.info(f"  Total time: {total_time:.2f}s")
    if sampling_enabled:
        logger.info(f"  Successful samples: {len(all_results)}/{len(flatten_datas)}")
        logger.info(f"  Total queries: {total_queries}")
    else:
        logger.info(f"  Successful queries: {len(all_results)}/{total_queries}")
    logger.info(f"  Average time per sample: {avg_time:.2f}s")
    logger.info(f"  Max time per sample: {max_time:.2f}s")
    logger.info(f"  Min time per sample: {min_time:.2f}s")
    logger.info(f"  Samples per second: {len(all_results)/total_time:.2f}")

    # Calculate statistics for each dataset & save results
    for dataset_id, queries_results in results_by_dataset_and_query.items():
        logger.info(f"Dataset {dataset_id} statistics:")
        
        if sampling_enabled:
            # Sampling mode: calculate best-of-n, average@n, pass@n
            best_of_n_accs = []
            average_at_n_accs = []
            pass_at_n_rates = []
            all_samples_for_dataset = []
            
            for query_idx, samples in queries_results.items():
                # Sort by sample_idx to ensure correct order
                samples = sorted(samples, key=lambda x: x.get("sample_idx", 0))
                all_samples_for_dataset.extend(samples)
                
                accuracies = [s['llm_as_judge']['accuracy'] for s in samples]
                
                # Best-of-n: take maximum accuracy
                best_of_n_accs.append(max(accuracies))
                
                # Average@n: average accuracy
                average_at_n_accs.append(np.mean(accuracies))
                
                # Pass@n: at least one correct
                pass_at_n_rates.append(1.0 if any(acc == 1.0 for acc in accuracies) else 0.0)
                
                # Save summary of all samples for this query
                summary_path = f"{config['exps']['output_dir']}/{dataset_id}/problem_{query_idx}_summary.json"
                best_idx = np.argmax(accuracies)
                summary = {
                    "query_idx": query_idx,
                    "question": samples[0]["question"],
                    "answer": samples[0]["answer"],
                    "n_samples": len(samples),
                    "metrics": {
                        "best_of_n": {
                            "accuracy": float(max(accuracies)),
                            "sample_idx": int(best_idx),
                            "prediction": samples[best_idx]["prediction"]
                        },
                        "average_at_n": float(np.mean(accuracies)),
                        "pass_at_n": float(1.0 if any(acc == 1.0 for acc in accuracies) else 0.0),
                        "all_accuracies": [float(acc) for acc in accuracies]
                    },
                    "processing_times": [s["processing_time_seconds"] for s in samples],
                    "total_processing_time": sum(s["processing_time_seconds"] for s in samples),
                    "sample_files": [f"problem_{query_idx}_sample_{i}.json" if i > 0 else f"problem_{query_idx}.json"
                                    for i in range(len(samples))]
                }
                FileUtils.save_json(summary, summary_path)
            
            # Summary statistics
            dataset_stats = {
                "best_of_n_accuracy": float(np.mean(best_of_n_accs)),
                "average_at_n_accuracy": float(np.mean(average_at_n_accs)),
                "pass_at_n_rate": float(np.mean(pass_at_n_rates)),
                "n_samples": n_samples,
                "successful_queries": len(queries_results),
                "total_queries": len(datas[dataset_id]),
                "success_rate": len(queries_results)/len(datas[dataset_id]) if len(datas[dataset_id]) > 0 else 0,
                "concurrency_limit": concurrency_limit,
                "max_workers": max_workers,
            }
            
            logger.info(f"  Successful queries: {len(queries_results)}/{len(datas[dataset_id])}")
            logger.info(f"  Best-of-{n_samples}: {dataset_stats['best_of_n_accuracy']:.4f}")
            logger.info(f"  Average@{n_samples}: {dataset_stats['average_at_n_accuracy']:.4f}")
            logger.info(f"  Pass@{n_samples}: {dataset_stats['pass_at_n_rate']:.4f}")
            
            # Save detailed results (including all samples)
            save_path = f"{config['exps']['output_dir']}/{dataset_id}/results.json"
            FileUtils.save_json({
                "statistics": dataset_stats,
                "results": all_samples_for_dataset,
            }, save_path)
            
            # Save concise accuracy summary
            FileUtils.save_json({
                "best_of_n_accuracy": dataset_stats["best_of_n_accuracy"],
                "average_at_n_accuracy": dataset_stats["average_at_n_accuracy"],
                "pass_at_n_rate": dataset_stats["pass_at_n_rate"],
                "n_samples": n_samples,
                "num_queries_with_correct_sample": int(sum(pass_at_n_rates)),
                "successful_queries": dataset_stats["successful_queries"],
                "total_queries": dataset_stats["total_queries"],
                "success_rate": dataset_stats["success_rate"],
            }, save_path.replace("results.json", "accuracy.json"))
            
            # Log to wandb
            if wandb_enabled:
                wandb.log({
                    f"{dataset_id}/best_of_n_accuracy": dataset_stats["best_of_n_accuracy"],
                    f"{dataset_id}/average_at_n_accuracy": dataset_stats["average_at_n_accuracy"],
                    f"{dataset_id}/pass_at_n_rate": dataset_stats["pass_at_n_rate"],
                    f"{dataset_id}/success_rate": dataset_stats["success_rate"],
                    f"{dataset_id}/successful_queries": dataset_stats["successful_queries"],
                    f"{dataset_id}/total_queries": dataset_stats["total_queries"],
                })
        else:
            # Original single trajectory statistics
            all_samples = [s for samples in queries_results.values() for s in samples]
            
            dataset_stats = {
                "accuracy": float(np.mean([s['llm_as_judge']['accuracy'] for s in all_samples])),
                "successful_queries": len(queries_results),
                "total_queries": len(datas[dataset_id]),
                "success_rate": len(queries_results)/len(datas[dataset_id]) if len(datas[dataset_id]) > 0 else 0,
                "concurrency_limit": concurrency_limit,
                "max_workers": max_workers,
            }
            
            logger.info(f"  Successful queries: {len(queries_results)}/{len(datas[dataset_id])}")
            logger.info(f"  Accuracy: {dataset_stats['accuracy']:.4f}")
            
            save_path = f"{config['exps']['output_dir']}/{dataset_id}/results.json"
            FileUtils.save_json({
                "statistics": dataset_stats,
                "results": all_samples,
            }, save_path)
            FileUtils.save_json({
                "accuracy": dataset_stats["accuracy"],
                "num_correct": len([item for item in all_samples if item['llm_as_judge']['accuracy'] == 1]),
                "num_incorrect": len([item for item in all_samples if item['llm_as_judge']['accuracy'] == 0]),
                "successful_queries": dataset_stats["successful_queries"],
                "total_queries": dataset_stats["total_queries"],
                "success_rate": dataset_stats["success_rate"],            
            }, save_path.replace("results.json", "accuracy.json"))
            
            # Log dataset results to wandb
            if wandb_enabled:
                wandb.log({
                    f"{dataset_id}/accuracy": dataset_stats["accuracy"],
                    f"{dataset_id}/success_rate": dataset_stats["success_rate"],
                    f"{dataset_id}/successful_queries": dataset_stats["successful_queries"],
                    f"{dataset_id}/total_queries": dataset_stats["total_queries"],
                })
        
        logger.info(f"  Dataset {dataset_id} results saved to {save_path}")
    if wandb_enabled:
        wandb.finish()

def _worker_process_batch(worker_args):
    """
    Worker function for subprocess: process a batch of data assigned to this process
    Use asyncio for async processing within the process
    """
    batch_data, config_path, threads_per_process, queue_multiplier, worker_id, override = worker_args
    
    # Re-acquire logger in subprocess
    worker_logger = get_logger(f"worker_{worker_id}")
    worker_logger.info(f"Worker {worker_id} started, processing {len(batch_data)} samples with {threads_per_process} async threads, queue_multiplier={queue_multiplier}")
    
    # Load config
    config = FileUtils.load_config(config_path)
    
    async def process_batch_async():
        """Asynchronously process a batch of data within subprocess, using limited queue to control active task count"""
        semaphore = asyncio.Semaphore(threads_per_process)
        max_active_tasks = threads_per_process * queue_multiplier
        
        worker_logger.info(f"Worker {worker_id}: Max active tasks = {max_active_tasks} (semaphore={threads_per_process} * multiplier={queue_multiplier})")
        
        async def process_single_query(data):
            """处理单个sample"""
            TIMEOUT = 10800  # 3 hours
            MAX_RETRIES = 4
            
            for attempt in range(MAX_RETRIES):
                try:
                    # First acquire semaphore (waiting time not counted in timeout)
                    async with semaphore:
                        # Start timing after acquiring semaphore (only count actual processing time)
                        async with asyncio.timeout(TIMEOUT):
                            # Create independent agent instance for each task
                            agent = eval(config["agent"]["cls"])(
                                config_path=config_path
                            )
                            
                            result = await agent.process_query(
                                idx=data["idx"],
                                user_query=data["problem"],
                                ground_truth=data["answer"],
                                dataset_id=data["dataset_id"],
                                sample_idx=data.get("sample_idx", 0),
                                override=override
                            )
                            
                            if result is not None:
                                result["sample_idx"] = data.get("sample_idx", 0)
                                result["n_samples"] = data.get("n_samples", 1)
                            
                            return result
                except asyncio.TimeoutError:
                    worker_logger.error(f"Worker {worker_id}: Query {data['idx']} sample {data.get('sample_idx', 0)} timed out (attempt {attempt + 1}/{MAX_RETRIES})")
                    if attempt == MAX_RETRIES - 1:
                        return None
                    await asyncio.sleep(5)
                except Exception as e:
                    worker_logger.error(f"Worker {worker_id}: Failed to process query {data['idx']} sample {data.get('sample_idx', 0)} (attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}: {e}")
                    worker_logger.error(traceback.format_exc())
                    if attempt == MAX_RETRIES - 1:
                        return None
                    await asyncio.sleep(5)
            
            return None
        
        # Use limited queue to control active task count, avoid too many different queries competing simultaneously causing prefix cache invalidation
        pending_data = list(batch_data)  # Data to be processed
        active_tasks = {}  # Active tasks: {task: data_info}
        results = []
        completed_count = 0
        
        # Initialize: fill queue to max_active_tasks
        while len(active_tasks) < max_active_tasks and pending_data:
            data = pending_data.pop(0)
            task = asyncio.create_task(process_single_query(data))
            active_tasks[task] = {"idx": data["idx"], "sample_idx": data.get("sample_idx", 0)}
        
        worker_logger.info(f"Worker {worker_id}: Initial queue filled with {len(active_tasks)} tasks, {len(pending_data)} tasks pending")
        
        # Continue processing until all tasks complete
        while active_tasks:
            # Wait for any task to complete
            done, pending_tasks = await asyncio.wait(
                active_tasks.keys(), 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Process completed tasks
            for task in done:
                task_info = active_tasks[task]
                completed_count += 1  # Count all completed tasks (regardless of success or failure)
                
                try:
                    result = await task
                    if result is not None and not isinstance(result, Exception):
                        results.append(result)
                except Exception as e:
                    worker_logger.error(f"Worker {worker_id}: Task for query {task_info['idx']} sample {task_info['sample_idx']} failed with exception: {e}")
                
                # Remove from active tasks
                del active_tasks[task]
            
            # Add new tasks to queue
            newly_added = 0
            while len(active_tasks) < max_active_tasks and pending_data:
                data = pending_data.pop(0)
                task = asyncio.create_task(process_single_query(data))
                active_tasks[task] = {"idx": data["idx"], "sample_idx": data.get("sample_idx", 0)}
                newly_added += 1
            
            # Periodically output progress
            if completed_count > 0 and completed_count % 10 == 0:
                worker_logger.info(f"Worker {worker_id}: Progress: {completed_count}/{len(batch_data)} completed, {len(active_tasks)} active, {len(pending_data)} pending")
        
        worker_logger.info(f"Worker {worker_id}: All tasks completed. Successful results: {len(results)}/{len(batch_data)}")
        return results
    
    # Create new event loop in subprocess and run
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(process_batch_async())
        loop.close()
        worker_logger.info(f"Worker {worker_id} finished, processed {len(results)}/{len(batch_data)} samples successfully")
        return results
    except Exception as e:
        worker_logger.error(f"Worker {worker_id} failed with error: {e}")
        worker_logger.error(traceback.format_exc())
        return []

def main_multiprocess_asyncio_multiple_files(args):
    """
    Multi-process + async coroutine version of main function
    - Multi-process level: distribute data by batch to different processes
    - Within process: use asyncio for async data processing
    - Data distribution: load balance between datasets, evenly distribute samples of each dataset to processes
    """
    config = FileUtils.load_config(args.config_path)
    exp_config = config["exps"]
    input_file_list = exp_config["input_file_list"]
    if not isinstance(input_file_list, list):
        input_file_list = [input_file_list]
    
    # Get config parameters
    num_processes = exp_config.get("num_processes", mp.cpu_count())
    threads_per_process = exp_config.get("threads_per_process", 8)
    queue_multiplier = exp_config.get("queue_multiplier", 2)  # Queue size = semaphore * queue_multiplier, used to control prefix cache
    
    start_time = time.time()
    FileUtils.save_json(config, f"{config['exps']['output_dir']}/config.json")
    
    # Initialize wandb if enabled (only in main process)
    wandb_enabled = config.get("wandb", {}).get("enabled", False)
    if wandb_enabled:
        os.environ["WANDB_API_KEY"] = config.get("wandb", {}).get("wandb_api_key", "")
        wandb_config = config.get("wandb", {})
        wandb.init(
            project=wandb_config.get("project", "deeptracehub"),
            name=wandb_config.get("run_name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            config={
                "agent_config": config.get("agent", {}),
                "llm_config": config.get("llm", {}),
                "tools_config": config.get("tools", {}),
                "exps_config": config.get("exps", {})
            },
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("notes", "")
        )
    
    # Read data from multiple files
    datas = {}
    for input_file in input_file_list:
        dataset_id = os.path.basename(input_file).split(".")[0]
        datas[dataset_id] = FileUtils.load_jsonl(input_file)
        logger.info(f"Loaded {len(datas[dataset_id])} queries from {input_file}")
    
    # Create output directory
    for dataset_id in datas.keys():
        if not os.path.exists(f"{config['exps']['output_dir']}/{dataset_id}"):
            os.makedirs(f"{config['exps']['output_dir']}/{dataset_id}", exist_ok=True)
            logger.info(f"Create directory: {config['exps']['output_dir']}/{dataset_id}")
    
    # Flatten data and expand samples according to sampling config
    # Use interleaving to ensure even distribution of samples from each dataset to different processes
    sampling_enabled = config["agent"].get("sampling", {}).get("enabled", False)
    n_samples = config["agent"]["sampling"]["n_samples"] if sampling_enabled else 1
    
    # First organize data by dataset
    dataset_samples = {}  # dataset_id -> list of samples
    for dataset_id, dataset_datas in datas.items():
        dataset_samples[dataset_id] = []
        for data in dataset_datas:
            if sampling_enabled:
                for sample_idx in range(n_samples):
                    sample_data = {
                        "dataset_id": dataset_id,
                        "idx": data["idx"],
                        "problem": data["problem"],
                        "answer": data["answer"],
                        "sample_idx": sample_idx,
                        "n_samples": n_samples
                    }
                    dataset_samples[dataset_id].append(sample_data)
            else:
                sample_data = {
                    "dataset_id": dataset_id,
                    "idx": data["idx"],
                    "problem": data["problem"],
                    "answer": data["answer"],
                    "sample_idx": 0,
                    "n_samples": 1
                }
                dataset_samples[dataset_id].append(sample_data)
    
    # Use interleaved distribution strategy to ensure even distribution of samples from each dataset to each process
    # Use round-robin: take one sample from each dataset in turn and put into flatten_datas
    flatten_datas = []
    output_dir = config['exps']['output_dir']
    
    # First build complete flatten_datas (using round-robin interleaved distribution)
    dataset_ids = list(dataset_samples.keys())
    indices = {did: 0 for did in dataset_ids}
    max_len = max(len(samples) for samples in dataset_samples.values())
    
    for i in range(max_len):
        for did in dataset_ids:
            if indices[did] < len(dataset_samples[did]):
                sample = dataset_samples[did][indices[did]]
                flatten_datas.append(sample)
                indices[did] += 1
    
    # Use efficient pre-filter function (batch get directory list + parallel read JSON)
    original_count = len(flatten_datas)
    flatten_datas, skipped_count, resume_count = _filter_completed_tasks(
        flatten_datas, output_dir, resume_mode=args.resume
    )
    logger.info(f"Pre-filtered: {original_count} -> {len(flatten_datas)} tasks ({skipped_count} skipped, {resume_count} to resume)")
    if sampling_enabled:
        logger.info(f"Flattened {len(flatten_datas)} tasks (samples) from {len(datas)} datasets")
        logger.info(f"Each query has {n_samples} samples, total queries: {sum(len(v) for v in datas.values())}")
    else:
        logger.info(f"Flattened {len(flatten_datas)} queries from {len(datas)} datasets")
    
    # Split data into num_processes batches
    batch_size = (len(flatten_datas) + num_processes - 1) // num_processes
    batches = []
    for i in range(num_processes):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(flatten_datas))
        if start_idx < len(flatten_datas):
            batches.append(flatten_datas[start_idx:end_idx])
    
    logger.info(f"Starting multiprocess + asyncio batch processing:")
    logger.info(f"  Total samples: {len(flatten_datas)}")
    logger.info(f"  Number of processes: {len(batches)}")
    logger.info(f"  Threads per process: {threads_per_process}")
    logger.info(f"  Queue multiplier: {queue_multiplier} (max active tasks per process: {threads_per_process * queue_multiplier})")
    logger.info(f"  Batch sizes: {[len(b) for b in batches]}")
    
    # Prepare worker arguments
    worker_args_list = [
        (batch, args.config_path, threads_per_process, queue_multiplier, worker_id, args.override)
        for worker_id, batch in enumerate(batches)
    ]
    
    # Execute using multiprocessing.Pool
    all_results = []
    try:
        with Pool(processes=len(batches)) as pool:
            logger.info("Submitting batches to process pool...")
            batch_results = pool.map(_worker_process_batch, worker_args_list)
            
            # Aggregate all results
            for results in batch_results:
                all_results.extend(results)
            
            logger.info(f"All workers finished. Total results: {len(all_results)}")
    except Exception as e:
        logger.error(f"Error in multiprocess execution: {e}")
        logger.error(traceback.format_exc())
        raise
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Group results by dataset_id and query_idx
    results_by_dataset_and_query = defaultdict(lambda: defaultdict(list))
    for result in all_results:
        dataset_id = result["dataset_id"]
        query_idx = result["idx"]
        results_by_dataset_and_query[dataset_id][query_idx].append(result)
    
    # Calculate outermost statistics
    processing_times = [result.get("processing_time_seconds", 0) for result in all_results]
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    max_time = max(processing_times) if processing_times else 0
    min_time = min(processing_times) if processing_times else 0
    
    total_queries = sum(len(v) for v in datas.values())
    logger.info(f"Multiprocess + asyncio batch processing completed:")
    logger.info(f"  Total time: {total_time:.2f}s")
    if sampling_enabled:
        logger.info(f"  Successful samples: {len(all_results)}/{len(flatten_datas)}")
        logger.info(f"  Total queries: {total_queries}")
    else:
        logger.info(f"  Successful queries: {len(all_results)}/{total_queries}")
    logger.info(f"  Average time per sample: {avg_time:.2f}s")
    logger.info(f"  Max time per sample: {max_time:.2f}s")
    logger.info(f"  Min time per sample: {min_time:.2f}s")
    logger.info(f"  Samples per second: {len(all_results)/total_time:.2f}")
    
    # Calculate statistics for each dataset & save results
    for dataset_id, queries_results in results_by_dataset_and_query.items():
        logger.info(f"Dataset {dataset_id} statistics:")
        
        if sampling_enabled:
            # Sampling mode: calculate best-of-n, average@n, pass@n
            best_of_n_accs = []
            average_at_n_accs = []
            pass_at_n_rates = []
            all_samples_for_dataset = []
            
            for query_idx, samples in queries_results.items():
                samples = sorted(samples, key=lambda x: x.get("sample_idx", 0))
                all_samples_for_dataset.extend(samples)
                
                accuracies = [s['llm_as_judge']['accuracy'] for s in samples]
                
                best_of_n_accs.append(max(accuracies))
                average_at_n_accs.append(np.mean(accuracies))
                pass_at_n_rates.append(1.0 if any(acc == 1.0 for acc in accuracies) else 0.0)
                
                # Save summary of all samples for this query
                summary_path = f"{config['exps']['output_dir']}/{dataset_id}/problem_{query_idx}_summary.json"
                best_idx = np.argmax(accuracies)
                summary = {
                    "query_idx": query_idx,
                    "question": samples[0]["question"],
                    "answer": samples[0]["answer"],
                    "n_samples": len(samples),
                    "metrics": {
                        "best_of_n": {
                            "accuracy": float(max(accuracies)),
                            "sample_idx": int(best_idx),
                            "prediction": samples[best_idx]["prediction"]
                        },
                        "average_at_n": float(np.mean(accuracies)),
                        "pass_at_n": float(1.0 if any(acc == 1.0 for acc in accuracies) else 0.0),
                        "all_accuracies": [float(acc) for acc in accuracies]
                    },
                    "processing_times": [s["processing_time_seconds"] for s in samples],
                    "total_processing_time": sum(s["processing_time_seconds"] for s in samples),
                    "sample_files": [f"problem_{query_idx}_sample_{i}.json" if i > 0 else f"problem_{query_idx}.json"
                                    for i in range(len(samples))]
                }
                FileUtils.save_json(summary, summary_path)
            
            dataset_stats = {
                "best_of_n_accuracy": float(np.mean(best_of_n_accs)),
                "average_at_n_accuracy": float(np.mean(average_at_n_accs)),
                "pass_at_n_rate": float(np.mean(pass_at_n_rates)),
                "n_samples": n_samples,
                "successful_queries": len(queries_results),
                "total_queries": len(datas[dataset_id]),
                "success_rate": len(queries_results)/len(datas[dataset_id]) if len(datas[dataset_id]) > 0 else 0,
                "num_processes": num_processes,
                "threads_per_process": threads_per_process,
            }
            
            logger.info(f"  Successful queries: {len(queries_results)}/{len(datas[dataset_id])}")
            logger.info(f"  Best-of-{n_samples}: {dataset_stats['best_of_n_accuracy']:.4f}")
            logger.info(f"  Average@{n_samples}: {dataset_stats['average_at_n_accuracy']:.4f}")
            logger.info(f"  Pass@{n_samples}: {dataset_stats['pass_at_n_rate']:.4f}")
            
            save_path = f"{config['exps']['output_dir']}/{dataset_id}/results.json"
            FileUtils.save_json({
                "statistics": dataset_stats,
                "results": all_samples_for_dataset,
            }, save_path)
            
            FileUtils.save_json({
                "best_of_n_accuracy": dataset_stats["best_of_n_accuracy"],
                "average_at_n_accuracy": dataset_stats["average_at_n_accuracy"],
                "pass_at_n_rate": dataset_stats["pass_at_n_rate"],
                "n_samples": n_samples,
                "num_queries_with_correct_sample": int(sum(pass_at_n_rates)),
                "successful_queries": dataset_stats["successful_queries"],
                "total_queries": dataset_stats["total_queries"],
                "success_rate": dataset_stats["success_rate"],
            }, save_path.replace("results.json", "accuracy.json"))
            
            if wandb_enabled:
                wandb.log({
                    f"{dataset_id}/best_of_n_accuracy": dataset_stats["best_of_n_accuracy"],
                    f"{dataset_id}/average_at_n_accuracy": dataset_stats["average_at_n_accuracy"],
                    f"{dataset_id}/pass_at_n_rate": dataset_stats["pass_at_n_rate"],
                    f"{dataset_id}/success_rate": dataset_stats["success_rate"],
                    f"{dataset_id}/successful_queries": dataset_stats["successful_queries"],
                    f"{dataset_id}/total_queries": dataset_stats["total_queries"],
                })
        else:
            all_samples = [s for samples in queries_results.values() for s in samples]
            
            dataset_stats = {
                "accuracy": float(np.mean([s['llm_as_judge']['accuracy'] for s in all_samples])),
                "successful_queries": len(queries_results),
                "total_queries": len(datas[dataset_id]),
                "success_rate": len(queries_results)/len(datas[dataset_id]) if len(datas[dataset_id]) > 0 else 0,
                "num_processes": num_processes,
                "threads_per_process": threads_per_process,
            }
            
            logger.info(f"  Successful queries: {len(queries_results)}/{len(datas[dataset_id])}")
            logger.info(f"  Accuracy: {dataset_stats['accuracy']:.4f}")
            
            save_path = f"{config['exps']['output_dir']}/{dataset_id}/results.json"
            FileUtils.save_json({
                "statistics": dataset_stats,
                "results": all_samples,
            }, save_path)
            FileUtils.save_json({
                "accuracy": dataset_stats["accuracy"],
                "num_correct": len([item for item in all_samples if item['llm_as_judge']['accuracy'] == 1]),
                "num_incorrect": len([item for item in all_samples if item['llm_as_judge']['accuracy'] == 0]),
                "successful_queries": dataset_stats["successful_queries"],
                "total_queries": dataset_stats["total_queries"],
                "success_rate": dataset_stats["success_rate"],
            }, save_path.replace("results.json", "accuracy.json"))
            
            if wandb_enabled:
                wandb.log({
                    f"{dataset_id}/accuracy": dataset_stats["accuracy"],
                    f"{dataset_id}/success_rate": dataset_stats["success_rate"],
                    f"{dataset_id}/successful_queries": dataset_stats["successful_queries"],
                    f"{dataset_id}/total_queries": dataset_stats["total_queries"],
                })
        
        logger.info(f"  Dataset {dataset_id} results saved to {save_path}")
    
    if wandb_enabled:
        wandb.finish()

async def main_asyncio(args):
    """Async version of main function, using asyncio.Semaphore to maintain a pool with capacity of num_processes"""
    config = FileUtils.load_config(args.config_path)
    exp_config = config["exps"]
    
    # Get config parameters
    num_processes = exp_config.get("num_processes", mp.cpu_count())
    max_workers = exp_config.get("max_workers", 512)
    
    # Calculate actual concurrency limit
    concurrency_limit = min(num_processes, max_workers)
    
    start_time = time.time()

    FileUtils.save_json(config, f"{config['exps']['output_dir']}/config.json")
    
    # Load data
    datas = FileUtils.load_jsonl(exp_config["input_file"])
    assert "idx" in datas[0] and "problem" in datas[0] and "answer" in datas[0], "Input file must contain 'idx', 'problem' and 'answer' fields"
    logger.info(f"Loaded {len(datas)} queries")
    
    # Add necessary fields to each data for using _filter_completed_tasks
    output_dir = config['exps']['output_dir']
    for data in datas:
        data["dataset_id"] = ""  # Single file mode has no dataset_id
        data["sample_idx"] = 0
    
    # Pre-filter completed tasks
    original_count = len(datas)
    datas, skipped_count, resume_count = _filter_completed_tasks(
        datas, output_dir, resume_mode=args.resume
    )
    logger.info(f"Pre-filtered: {original_count} -> {len(datas)} tasks ({skipped_count} skipped, {resume_count} to resume)")
    
    # Create shared key_manager
    key_manager = SerperDevKeyManager(config["tools"]["serper_key_usage_file"])
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def process_single_query(data):
        """Async function to process single query"""
        TIMEOUT = 6000
        # First acquire semaphore (waiting time not counted in timeout)
        async with semaphore:  # Acquire semaphore to limit concurrency
            try:
                # Start timing after acquiring semaphore (only count actual processing time)
                async with asyncio.timeout(TIMEOUT):
                    # Create independent agent instance for each task
                    agent = eval(config["agent"]["cls"])(
                        config_path=args.config_path,
                        key_manager=key_manager
                    )
                    
                    result = await agent.process_query(
                        idx=data["idx"], 
                        user_query=data["problem"], 
                        ground_truth=data["answer"],
                        override=args.override
                    )
                    return result
            except asyncio.TimeoutError:
                logger.error(f"Query {data['idx']} timed out after {TIMEOUT} seconds")
                logger.error(traceback.format_exc())
                return None
            except Exception as e:
                logger.error(f"Failed to process query {data['idx']}: {e}")
                logger.error(traceback.format_exc())
                return None

    
    logger.info(f"Starting asyncio batch processing:")
    logger.info(f"  Total queries: {len(datas)}")
    logger.info(f"  Concurrency limit: {concurrency_limit}")
    logger.info(f"  Max workers: {max_workers}")
    
    # Use asyncio.gather to execute all tasks concurrently
    all_results = []
    
    try:
        logger.info("Submitting tasks to asyncio...")
        # 创建所有任务
        tasks = [process_single_query(data) for data in datas]
        
        # 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        for result in results:
            if result is not None:
                all_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                
    except Exception as e:
        logger.error(f"Error in asyncio execution: {e}")
        raise
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    processing_times = [result.get("processing_time_seconds", 0) for result in all_results]
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    max_time = max(processing_times) if processing_times else 0
    min_time = min(processing_times) if processing_times else 0
    
    logger.info(f"Asyncio batch processing completed:")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Successful queries: {len(all_results)}/{len(datas)}")
    logger.info(f"  Average time per query: {avg_time:.2f}s")
    logger.info(f"  Max time per query: {max_time:.2f}s")
    logger.info(f"  Min time per query: {min_time:.2f}s")
    logger.info(f"  Queries per second: {len(datas)/total_time:.2f}")
    
    # Save results
    results = {
        "statistics": {
            "total_time": total_time,
            "avg_time": avg_time,
            "max_time": max_time,
            "min_time": min_time,
            "accuracy": np.mean([item["llm_as_judge"]["accuracy"] for item in all_results]),
            "queries_per_second": len(datas)/total_time,
            "successful_queries": len(all_results),
            "total_queries": len(datas),
            "success_rate": len(all_results)/len(datas) if len(datas) > 0 else 0,
            "concurrency_limit": concurrency_limit,
            "max_workers": max_workers
        },
        "results": all_results,
    }
    
    FileUtils.save_json(results, f"{config['exps']['output_dir']}/results.json")
    logger.info(f"Accuracy: {results['statistics']['accuracy']}")
    logger.info(f"Results saved to {config['exps']['output_dir']}/results.json")

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    config = FileUtils.load_config(args.config_path)

    if not os.path.exists(f"{config['exps']['output_dir']}"):
        os.makedirs(f"{config['exps']['output_dir']}", exist_ok=True)
    logger = get_logger(__name__)

    logger.info(f"Starting {args.config_path}")
    logger.info(f"Config: {config}")

    if args.debug:
        main_debug(args.config_path)
    else:
        if config["exps"].get("input_file_list", None) is not None:
            if args.multiprocess:
                logger.info(f"Running multiprocess + asyncio multiple files")
                main_multiprocess_asyncio_multiple_files(args)
            else:
                logger.info(f"Running asyncio multiple files")
                asyncio.run(main_asyncio_multiple_files(args))
        else:   
            logger.info(f"Running asyncio")
            asyncio.run(main_asyncio(args))
