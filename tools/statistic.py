import json
import argparse
import numpy as np
import os
import re

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Input folder")
    return parser.parse_args()

def get_all_files(folder_path):
    """Get all files in problem_{idx}.json format under the folder"""
    all_files = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist")
        return all_files
    
    if not os.path.isdir(folder_path):
        print(f"Error: '{folder_path}' is not a folder")
        return all_files
    
    # Define regex to match problem_{idx}.json
    pattern = re.compile(r'^problem_\d+\.json$')
    
    # Walk through the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if filename matches problem_{idx}.json format
            if pattern.match(file):
                file_path = os.path.join(root, file)
                all_files.append(file_path)
    
    return all_files

def func_get_acc(file_path):
    data = json.load(open(file_path, "r"))
    acc = data["llm_as_judge"]["accuracy"]
    return acc

if __name__ == "__main__":
    args = parse_args()
    input_folder = args.input
    
    if not input_folder:
        print("Please use --input parameter to specify the folder path")
        exit(1)
    
    # Get all files matching the format
    files = get_all_files(input_folder)
    
    print(f"Found {len(files)} files matching problem_{{idx}}.json format:")
    results = list(map(func_get_acc, files))
    num_correct = sum(results)
    print(f"Accuracy: {np.mean(results)} ({num_correct}/{len(files)})")
    with open(f"{input_folder}/accuracy.json", "w") as f:
        json.dump({"accuracy": np.mean(results), "num_correct": num_correct, "total": len(results)}, f, indent=4, ensure_ascii=False)
