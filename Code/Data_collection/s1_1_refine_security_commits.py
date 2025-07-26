import pickle
import os
import re
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import argparse
from datetime import datetime
from s1_1_utils import *  # Assumed utility functions for splitting diffs, extracting functions, etc.

def catch_commit_datetime(commit_message):
    """
    Extract and return the commit year from a git‑show output.

    Args:
        commit_message (str): Full commit text including the "Date: ..." line.

    Returns:
        int: Year of the commit, or None if no date found.
    """
    date_pattern = r"Date:\s+(.*)"
    match = re.search(date_pattern, commit_message)
    if not match:
        return None

    date_str = match.group(1)
    # Parse into datetime object, then return the year
    commit_date = datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y %z")
    return commit_date.year


def s1_filter_commit_list(commit_list):
    """
    Refine the list of LLM‑approved security commits by deeper code analysis.

    Steps:
    1. Attach full pre‑fix and post‑fix file contents (multi_add_prefix_fix_content).
    2. Remove commits where extracted prefix and fix functions are identical (no real change).
    3. Extract only the function block corresponding to the single diff hunk.
    4. Keep only commits that modify exactly one block.
    5. Discard commits whose prefix code is too large (>1000 tokens).
    6. Remove commits that only add lines (no deletions).
    7. Keep only commits dated before 2023.

    Args:
        commit_list (List[tuple]): Each entry is
            (repo, sha, full_diff, changed_file, prefix_content, fix_content).

    Returns:
        List[tuple]: Filtered and refined commit entries.
    """
    num_workers = cpu_count()

    # 1. Enrich each commit with its full prefix and fix file contents in parallel
    commit_list = multi_add_prefix_fix_content(commit_list, num_workers)

    def remove_noop_commits(commits):
        """
        Remove commits where the function before and after fix are identical (no real change).
        """
        kept = []
        for entry in commits:
            repo, sha, full_diff, changed_file, prefix_content, fix_content = entry
            meta_info, code_diffs = split_commit_content(full_diff)
            hunk_headers = extract_hunks(code_diffs[changed_file])
            diff_blocks = split_diff_by_blocks(code_diffs[changed_file])

            # Check each block: if prefix/fix function bodies match, drop commit
            noop = False
            for idx, (block_header, block_body) in enumerate(diff_blocks):
                header = hunk_headers[idx]
                before_fn = extract_function(prefix_content, header)
                after_fn  = extract_function(fix_content, header)
                if before_fn == after_fn:
                    noop = True
                    break
            if not noop:
                kept.append(entry)

        return kept

    # 2. Remove false‑positives with no actual code change
    commit_list = remove_noop_commits(commit_list)

    # 3. Extract only the one function block related to the diff
    commit_list = extract_prefix_block_function(commit_list)

    # 4. Keep only commits that change a single block (hunk)
    commit_list = multi_extract_single_block_content(commit_list, num_workers)

    # 5. Filter out commits whose prefix function is too large (>1000 tokens)
    prefix_texts = [entry[-2] for entry in commit_list]
    lengths = compute_prompts(prefix_texts)
    small_prefix_commits = [
        commit_list[i] for i, length in enumerate(lengths) if length < 1000
    ]

    # 6. Remove commits that only add lines (no deletions)
    with_deletions = [
        c for c in small_prefix_commits if not check_only_add_commit(c[2])
    ]

    # 7. Keep only commits before 2023
    final_commits = [
        c for c in with_deletions
        if catch_commit_datetime(c[2]) is not None and catch_commit_datetime(c[2]) < 2023
    ]

    return final_commits


def main(reponame):
    """
    Load stage‑0 and stage‑1 data, apply refinement filters, and save updated list.

    Args:
        reponame (str): Repository identifier (e.g. "FFmpeg-FFmpeg" or "Android/kernel-common").
    """
    # Paths for input/output
    base_s0 = f'../../automated_data/repo/{reponame}/s0'
    base_s1 = f'../../automated_data/repo/{reponame}/s1'
    os.makedirs(base_s1, exist_ok=True)

    # 1. Load commits already filtered by keyword and LLM judgment
    input_commits = pickle.load(open(os.path.join(base_s0, 's0_2_security_commits_filtered.pkl'), 'rb'))
    llm_results_path = os.path.join(base_s0, 's0_3_commit_security_analysis_results.pkl')
    llm_results = pickle.load(open(llm_results_path, 'rb'))

    # Ensure alignment between commits and LLM outputs
    assert len(input_commits) == len(llm_results)

    # 2. Select only those LLM‑labeled “yes”
    yes_idx = [i for i, res in enumerate(llm_results) if 'yes' in res.lower()]
    yes_commits = [input_commits[i] for i in yes_idx]

    # 3. Apply detailed filtering pipeline
    refined = s1_filter_commit_list(yes_commits)

    # 4. Save the refined commit list for the next stage
    output_path = os.path.join(base_s1, 's1_1_security_commits_updated.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(refined, f)

    print(f"Refined {len(refined)} security commits saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage s1_1: refine LLM‑approved security commits')
    parser.add_argument(
        '--reponame', type=str, required=True,
        help='Repository name, prefix Android/ for Android repos.'
    )
    args = parser.parse_args()
    main(args.reponame)
