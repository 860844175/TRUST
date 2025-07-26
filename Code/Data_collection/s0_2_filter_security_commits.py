import pickle
import os
import subprocess
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
import re
import argparse
from ..s0_utils import *

"""
This script continues from the previous step:
1. Load security-related commits identified by keywords.
2. Filter commits based on length, file type, and number of changed files.
3. Save the filtered commits for downstream processing.
"""

def filter_commit_list(commit_list, max_token_length=1000, only_c=True, single_file=True):
    """
    Filter a list of commits according to several criteria.

    Args:
        commit_list (List[tuple]): List of (repo, sha, diff text) tuples.
        max_token_length (int): Maximum allowed token length for the diff text.
        only_c (bool): If True, keep only commits that modify C/C++ files.
        single_file (bool): If True, keep only commits that touch exactly one file.

    Returns:
        List[tuple]: Filtered list of commits with an added filename element.
    """
    # Compute token lengths for each commit diff
    commit_length_list = compute_prompts([entry[2] for entry in commit_list])
    
    # Keep only commits under the length threshold
    short_commits = [
        commit_list[idx]
        for idx, length in enumerate(commit_length_list)
        if length < max_token_length
    ]
    print(f"After length filter: {len(short_commits)} commits remain.")
    
    # Determine changed filenames in parallel
    with mp.Pool(mp.cpu_count()) as pool:
        changed_files = list(
            tqdm(pool.imap(get_change_file_name_by_text, [c[2] for c in short_commits]),
                 total=len(short_commits),
                 desc="Detecting changed files")
        )
    
    # Keep commits that change exactly one file
    single_file_commits = [
        (repo, sha, diff, changed_files[idx][0])
        for idx, (repo, sha, diff) in enumerate(short_commits)
        if len(changed_files[idx]) == 1
    ]
    print(f"After single-file filter: {len(single_file_commits)} commits remain.")
    
    # If only C/C++ files are requested, further filter by extension
    if only_c:
        c_file_commits = [
            entry for entry in single_file_commits
            if entry[3].endswith(('.c', '.cc', '.cpp', '.h'))
        ]
        print(f"After C-file filter: {len(c_file_commits)} commits remain.")
        return c_file_commits
    
    return single_file_commits


def main(reponame):
    """
    Main procedure: load prior results, apply filters, and save output.

    Args:
        reponame (str): Name of the target repository folder. 
                         For Android repos, prefix with 'Android/'.
    """
    # Load commits matched by security keywords from stage 0
    input_path = f'../../automated_data/repo/{reponame}/s0/s0_0_keyword_match_commits.pkl'
    commits = pickle.load(open(input_path, 'rb'))
    
    # Drop any extremely large diffs to avoid memory issues
    commits = [c for c in commits if len(c[2]) < 1_000_000]
    
    # Apply filtering pipeline
    filtered = filter_commit_list(commits)
    print(f"Total filtered commits: {len(filtered)}")
    
    # Save filtered commits for stage 2
    output_path = f'../../automated_data/repo/{reponame}/s0/s0_2_security_commits_filtered.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(filtered, f)
    print(f"Filtered commits saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter security-related commit list')
    parser.add_argument(
        '--reponame', type=str, required=True,
        help='Repository name (e.g., FFmpeg-FFmpeg). For Android, prefix with Android/.'
    )
    args = parser.parse_args()
    main(args.reponame)

import json
import pickle
from vllm import LLM, SamplingParams
import os
import tiktoken
from tqdm import tqdm
import argparse
from glob import glob

"""
This script continues from s0_2 and performs step s0_3:
Use a large language model (LLM) to judge whether each previously filtered commit 
actually fixes a security vulnerability.
"""

# System prompt guiding the LLM through a detailed vulnerability analysis workflow
system_prompt = """You are a security expert specializing in vulnerability analysis. Your task is to analyze a given commit, including its message and code diff, to determine if the commit addresses a security vulnerability exists in prefix code. Follow the steps below for your analysis.

... (omitted for brevity, same as above) ...
"""

# Template for user message wrapping the commit content
user_message = """
I am providing you with a commit message and the corresponding code diff. Your task is to analyze whether this commit is fixing a security vulnerability based on the following step-by-step process:

1. Check if the commit is a merge commit or if there is no code diff. If either is true, return **Answer: no**.
2. If not, proceed to analyze the commit message for security-related keywords.
3. Examine the code changes based on:
   - Variable Type Analysis
   - Pointer Handling
   - Buffer Management
   - Memory Management
   - Permission and Access Control
   - User Input Validation
   - Data Integrity and Overflow Protection
   - Race Condition Prevention
4. Provide your answer in the following format: **Answer: [yes, no, cannot decide]**.

Here is the commit information:

#### Commit Content:
{}
"""

# Full prompt combining system and user messages for each commit
prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def main(input_dataset, output_path, start, end):
    """
    Process a slice of the input commit dataset by sending each commit to the LLM,
    collecting the LLM's security-fix judgement, and saving the results.

    Args:
        input_dataset (List[tuple]): List of (repo, sha, diff text) tuples.
        output_path (str): Directory in which to save output pickle files.
        start (int): Starting index (inclusive) for this batch.
        end (int): Ending index (exclusive) for this batch.
    """
    print(f"Processing commits {start} to {end}...")

    # Build prompt for each commit
    batch_prompts = []
    for repo, sha, diff_text in input_dataset[start:end]:
        # Insert the raw diff text into the user message
        user_filled = user_message.format(diff_text)
        # Combine with the system prompt
        full_prompt = prompt_template.format(system=system_prompt, user=user_filled)
        batch_prompts.append(full_prompt)

    # Initialize the LLM with meta-llama model (adjust tensor parallelism as needed)
    model_id = "meta-llama/Llama-3.1-70B-Instruct"
    llm = LLM(
        model=model_id,
        tensor_parallel_size=4,
        disable_custom_all_reduce=True,
        download_dir="/home/jyu7/"
    )
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=20480)

    # Generate LLM outputs in a batch
    print("Calling LLM for batch inference...")
    outputs = llm.generate(batch_prompts, sampling_params)
    results = [out.outputs[0].text.strip() for out in outputs]

    # Determine output filename based on batch indices
    if start == 0 and end == len(input_dataset):
        filename = 's0_3_commit_security_analysis_results.pkl'
    else:
        filename = f's0_3_commit_security_analysis_results_{start}_{end}.pkl'
    save_path = os.path.join(output_path, filename)

    # Save results to disk
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved analysis results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Security commit analysis using LLM')
    parser.add_argument(
        '--reponame', type=str, required=True,
        help='Name of the repository (same as used in previous stages)'
    )
    parser.add_argument(
        '--start', type=int, default=None,
        help='Start index of the commit slice (inclusive)'
    )
    parser.add_argument(
        '--end', type=int, default=None,
        help='End index of the commit slice (exclusive)'
    )
    args = parser.parse_args()

    # Construct paths based on the repository name
    input_path = f'../../automated_data/repo/{args.reponame}/s0/s0_2_security_commits_filtered.pkl'
    print(f"Loading filtered commits from {input_path}")
    commits = pickle.load(open(input_path, 'rb'))

    # Determine slice boundaries
    start_idx = args.start if args.start is not None else 0
    end_idx = args.end if args.end is not None else len(commits)

    # Ensure output directory exists
    output_dir = os.path.dirname(input_path)

    # Run main processing
    main(commits, output_dir, start_idx, end_idx)