import re
import os
import sys
import json
import json_repair
import time
import math
import wandb
import json5
import random
import asyncio
import logging
import tiktoken
import argparse
import traceback
import numpy as np
import multiprocessing as mp
from copy import deepcopy
from dotenv import load_dotenv
from typing import Any, Dict, List, Tuple, Union
from pathlib import Path
from datetime import datetime
from functools import partial
from collections import defaultdict
from transformers import AutoTokenizer
from multiprocessing import Pool, Manager
from tqdm import tqdm
if os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2]) not in sys.path:
    base_path = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2])
    sys.path.append(base_path)

from src.llms_async import async_chat_completion
from utils.file_utils import FileUtils, logger
from tools.judge import async_local_judge as llm_as_judge
logger = logging.getLogger(__name__)



class DirectQAVerifier:
    def __init__(self, config_path, **kwargs):
        self.config = FileUtils.load_config(config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["verifier"]["tokenizer"])
        self.max_tokens = self.config["verifier"]["max_tokens"]
        self.n_samples = self.config["exps"]["n_samples"]
        self._init_llm_endpoint()

    def _init_llm_endpoint(self):
        # Used for sticky requests to vLLM engine to maximize utilization of prefix-caching
        if self.config["verifier"]["endpoint_list"] is not None and len(self.config["verifier"]["endpoint_list"]) > 0:
            self.llm_endpoint = random.choice(self.config["verifier"]["endpoint_list"])
        else:
            self.llm_endpoint = self.config["verifier"]["endpoint"]
    
    def extract_answer(self, content: str):
        answer_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        else:
            return content[-512:]

    async def verify(self, idx: int, user_query: str, ground_truth: str, dag_file: str = None):
        try:
            # Ensure problems directory exists
            problems_dir = f"{self.config['exps']['output_dir']}/problems"
            os.makedirs(problems_dir, exist_ok=True)
            save_path = f"{problems_dir}/problem_{idx}.json"
            system_prompt = "\nAfter reasoning, use <answer> </answer> tags to wrap your reasoning summary andfinal answer."
            messages = [
                {"role": "user", "content": user_query + system_prompt}
            ]
            contents = await async_chat_completion(messages, self.config["verifier"]["model_name"], self.llm_endpoint, n=self.n_samples)
            extracted_answers = [self.extract_answer(content) for content in contents]
            judge_results = await asyncio.gather(*[
                llm_as_judge(
                    question=user_query,
                    response=ext_ans,
                    correct_answer=ground_truth,
                    endpoint=self.config["judge"]["local"]["endpoint"],
                    model_name=self.config["judge"]["local"]["model_name"]
                ) for ext_ans in extracted_answers
            ])
            accuracies = [item["accuracy"] for item in judge_results]
            avg_accuracy = np.mean(accuracies)
            pass_at_n = max(accuracies)
            dumps_data = {
                "idx": idx,
                "problem": user_query,
                "answer": ground_truth,
                "dag_file": dag_file,
                "reasoning": contents,
                "prediction": extracted_answers,
                "llm_as_judge": judge_results,
                "metrics": {
                    "accuracy": accuracies,
                    "avg_accuracy": avg_accuracy,
                    "pass_at_n": pass_at_n
                }
            }
            with open(save_path, "w") as f:
                json.dump(dumps_data, f, indent=4, ensure_ascii=False)
            return dumps_data
        except Exception as e:
            logger.error(f"Failed to verify: {e}")
            logger.error(traceback.format_exc())
            return None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./config.yaml")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--override", action="store_true", default=False)
    parser.add_argument("--multiprocess", action="store_true", default=False,
                        help="Use multiprocess + asyncio mode for better parallelism")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume incomplete tasks (files with 'resume': true)")
    return parser.parse_args()

def _worker_process_batch(worker_args):
    """
    Worker function for subprocess: process a batch of data assigned to this process
    Use asyncio for async processing within the process
    """
    batch_data, config_path, threads_per_process, queue_multiplier, worker_id = worker_args
    
    # Re-acquire logger in subprocess
    worker_logger = logging.getLogger(f"worker_{worker_id}")
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
                            # Create independent verifier instance for each task
                            verifier = DirectQAVerifier(config_path=config_path)
                            
                            result = await verifier.verify(
                                idx=data["idx"],
                                user_query=data["problem"],
                                ground_truth=data["answer"],
                                dag_file=data.get("dag_file", "")
                            )
                            
                            if result is not None:
                                result["dataset_id"] = data.get("dataset_id", "")
                            
                            return result
                except asyncio.TimeoutError:
                    worker_logger.error(f"Worker {worker_id}: Query {data['idx']} timed out (attempt {attempt + 1}/{MAX_RETRIES})")
                    if attempt == MAX_RETRIES - 1:
                        return None
                    await asyncio.sleep(5)
                except Exception as e:
                    worker_logger.error(f"Worker {worker_id}: Failed to process query {data['idx']} (attempt {attempt + 1}/{MAX_RETRIES}): {type(e).__name__}: {e}")
                    worker_logger.error(traceback.format_exc())
                    if attempt == MAX_RETRIES - 1:
                        return None
                    await asyncio.sleep(5)
            
            return None
        
        # Use limited queue to control active task count
        pending_data = list(batch_data)  # Data to be processed
        active_tasks = {}  # Active tasks: {task: data_info}
        results = []
        completed_count = 0
        
        # Initialize: fill queue to max_active_tasks
        while len(active_tasks) < max_active_tasks and pending_data:
            data = pending_data.pop(0)
            task = asyncio.create_task(process_single_query(data))
            active_tasks[task] = {"idx": data["idx"]}
        
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
                completed_count += 1
                
                try:
                    result = await task
                    if result is not None and not isinstance(result, Exception):
                        results.append(result)
                except Exception as e:
                    worker_logger.error(f"Worker {worker_id}: Task for query {task_info['idx']} failed with exception: {e}")
                
                # Remove from active tasks
                del active_tasks[task]
            
            # Add new tasks to queue
            while len(active_tasks) < max_active_tasks and pending_data:
                data = pending_data.pop(0)
                task = asyncio.create_task(process_single_query(data))
                active_tasks[task] = {"idx": data["idx"]}
            
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
    """
    config = FileUtils.load_config(args.config_path)
    exp_config = config["exps"]
    input_file_list = exp_config.get("input_file_list", [])
    if not isinstance(input_file_list, list):
        input_file_list = [input_file_list]
    
    # Get config parameters
    num_processes = exp_config.get("num_processes", mp.cpu_count())
    threads_per_process = exp_config.get("threads_per_process", 8)
    queue_multiplier = exp_config.get("queue_multiplier", 2)
    
    start_time = time.time()
    FileUtils.save_json(config, f"{config['exps']['output_dir']}/config.json")
    
    # Read data from multiple files
    datas = {}
    for input_file in input_file_list:
        if input_file.endswith('.jsonl'):
            dataset_id = os.path.basename(input_file).replace('.jsonl', '')
            datas[dataset_id] = FileUtils.load_jsonl(input_file)
        elif input_file.endswith('.json'):
            dataset_id = os.path.basename(input_file).replace('.json', '')
            with open(input_file, 'r') as f:
                data_list = json.load(f)
                if isinstance(data_list, list):
                    datas[dataset_id] = data_list
                else:
                    logger.error(f"Input file {input_file} should contain a list")
                    continue
        logger.info(f"Loaded {len(datas[dataset_id])} queries from {input_file}")
    
    # Create output directory
    output_dir = config['exps']['output_dir']
    problems_dir = f"{output_dir}/problems"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(problems_dir):
        os.makedirs(problems_dir, exist_ok=True)
    
    # Get all processed file lists at once
    logger.info("Checking existing files...")
    existing_files = set()
    if os.path.exists(problems_dir):
        try:
            # Use os.listdir to get all files at once
            all_files = os.listdir(problems_dir)
            # 提取已处理的 problem idx
            for filename in all_files:
                if filename.startswith('problem_') and filename.endswith('.json'):
                    # Extract '123' from 'problem_123.json'
                    try:
                        idx_str = filename.replace('problem_', '').replace('.json', '')
                        idx = int(idx_str)
                        existing_files.add(idx)
                    except ValueError:
                        # If filename format is incorrect, skip
                        logger.warning(f"Skipping invalid filename: {filename}")
                        continue
            logger.info(f"Found {len(existing_files)} existing files")
        except Exception as e:
            logger.warning(f"Failed to list directory {problems_dir}: {e}")
    
    # Flatten data and filter processed files
    flatten_datas = []
    skipped_count = 0
    
    for dataset_id, dataset_datas in datas.items():
        for data in dataset_datas:
            idx = data['idx']
            
            # Check if already processed (unless using --override)
            if idx in existing_files and not args.override:
                skipped_count += 1
                continue
            
            sample_data = {
                "dataset_id": dataset_id,
                "idx": idx,
                "problem": data["problem"],
                "answer": data["answer"],
                "dag_file": data.get("dag_file", "")
            }
            flatten_datas.append(sample_data)
    
    logger.info(f"Skipped {skipped_count} existing files")
    logger.info(f"Total {len(flatten_datas)} tasks to process")
    
    if len(flatten_datas) == 0:
        logger.info("No tasks to process. Exiting.")
        return
    
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
    logger.info(f"  Queue multiplier: {queue_multiplier}")
    logger.info(f"  Batch sizes: {[len(b) for b in batches]}")
    
    # Prepare worker arguments
    worker_args_list = [
        (batch, args.config_path, threads_per_process, queue_multiplier, worker_id)
        for worker_id, batch in enumerate(batches)
    ]
    
    # Execute using multiprocessing.Pool, and add progress bar
    all_results = []
    try:
        with Pool(processes=len(batches)) as pool:
            logger.info("Submitting batches to process pool...")
            logger.info("Progress bar will update as each batch completes...")
            
            # Use imap_unordered to get results one by one, for real-time progress bar updates
            # chunksize=1 ensures each batch returns immediately after completion, no buffering
            with tqdm(
                total=len(flatten_datas), 
                desc="Processing queries", 
                unit="query",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                mininterval=0.1  # Minimum update interval to avoid too frequent updates
            ) as pbar:
                completed_batches = 0
                for batch_result in pool.imap_unordered(_worker_process_batch, worker_args_list, chunksize=1):
                    # Update completed query count
                    num_results_in_batch = len(batch_result)
                    all_results.extend(batch_result)
                    completed_batches += 1
                    pbar.update(num_results_in_batch)
                    
                    # Update progress bar postfix info
                    pbar.set_postfix({
                        'batches': f'{completed_batches}/{len(batches)}',
                        'success': len(all_results)
                    })
                    
                    # Force refresh progress bar display
                    pbar.refresh()
            
            logger.info(f"All workers finished. Total results: {len(all_results)}/{len(flatten_datas)}")
    except Exception as e:
        logger.error(f"Error in multiprocess execution: {e}")
        logger.error(traceback.format_exc())
        raise
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    if len(all_results) > 0:
        # Calculate average accuracy and pass@n
        all_avg_accuracies = [result["metrics"]["avg_accuracy"] for result in all_results]
        all_pass_at_n = [result["metrics"]["pass_at_n"] for result in all_results]
        
        overall_stats = {
            "total_time": total_time,
            "successful_queries": len(all_results),
            "total_queries": len(flatten_datas) + skipped_count,
            "success_rate": len(all_results) / (len(flatten_datas) + skipped_count) if (len(flatten_datas) + skipped_count) > 0 else 0,
            "avg_accuracy": float(np.mean(all_avg_accuracies)),
            "pass_at_n": float(np.mean(all_pass_at_n)),
            "queries_per_second": len(all_results) / total_time if total_time > 0 else 0,
            "num_processes": num_processes,
            "threads_per_process": threads_per_process
        }
        
        logger.info(f"Multiprocess + asyncio batch processing completed:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Successful queries: {len(all_results)}/{len(flatten_datas) + skipped_count}")
        logger.info(f"  Average accuracy: {overall_stats['avg_accuracy']:.4f}")
        logger.info(f"  Pass@N: {overall_stats['pass_at_n']:.4f}")
        logger.info(f"  Queries per second: {overall_stats['queries_per_second']:.2f}")
        
        # Save results
        results_data = {
            "statistics": overall_stats,
            "results": all_results
        }
        
        FileUtils.save_json(results_data, f"{config['exps']['output_dir']}/results.json")
        FileUtils.save_json({
            "avg_accuracy": overall_stats["avg_accuracy"],
            "pass_at_n": overall_stats["pass_at_n"],
            "successful_queries": overall_stats["successful_queries"],
            "total_queries": overall_stats["total_queries"],
            "success_rate": overall_stats["success_rate"]
        }, f"{config['exps']['output_dir']}/accuracy.json")
        
        logger.info(f"Results saved to {config['exps']['output_dir']}/results.json")
    else:
        logger.warning("No results to save")


if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    config = FileUtils.load_config(args.config_path)
    
    if not os.path.exists(config['exps']['output_dir']):
        os.makedirs(config['exps']['output_dir'], exist_ok=True)
    
    logger.info(f"Starting {args.config_path}")
    logger.info(f"Config: {config}")

    main_multiprocess_asyncio_multiple_files(args)