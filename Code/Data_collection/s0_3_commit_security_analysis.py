import os
import pickle
import subprocess
from tqdm import tqdm
import multiprocessing as mp
import re
import argparse
from datetime import datetime

# List of security-related actions to search for in commit messages
def SECURITY_ACTIONS():
    return [
        'fix', 'patch', 'repair',
        'mitigate', 'prevent', 'protect',
        'secure', 'security', 'safeguard',
        'sanitize', 'validate', 'check',
        'audit', 'review',
        'block', 'filter',
        'exploit', 'attack', 'threat',
        'crash', 'corruption',
        'bypass', 'break',
        'vulnerable', 'vulnerability', 'vuln',
        'cve', 'advisory', 'security-issue'
    ]

# List of vulnerability types (e.g., memory issues, injection flaws)
def VULNERABILITY_TYPES():
    return [
        'overflow', 'underflow',
        'memory', 'free', 'malloc', 'leak', 'uaf', 'use-after-free',
        'heap', 'stack',
        'buffer', 'bounds',
        'null', 'nullptr', 'null-pointer',
        'race', 'deadlock', 'concurrency',
        'injection', 'sql', 'xss', 'csrf',
        'integer', 'divide',
        'format', 'string',
        'privilege', 'permission',
        'sandbox', 'escape'
    ]

# General problem descriptions indicating bugs or issues
def PROBLEM_DESCRIPTIONS():
    return [
        'bug', 'issue', 'defect',
        'unsafe', 'insecure', 'dangerous',
        'missing', 'improper', 'incorrect',
        'broken', 'invalid', 'malicious',
        'unauthorized', 'unauth',
        'compromise', 'breach',
        'critical', 'severe'
    ]

# Combine all security-related keywords into one list
SECURITY_RELATED_KEYWORDS = (
    VULNERABILITY_TYPES() +
    SECURITY_ACTIONS() +
    PROBLEM_DESCRIPTIONS()
)


def split_commit_content(retrieved_content: str):
    """
    Split git log output into individual commits.

    Args:
        retrieved_content (str): Output from `git log --grep` command.

    Returns:
        List[str]: Each element is a single commit's full text.
    """
    pattern = r'(commit [0-9a-f]{40}\n(?:.|\n)*?)(?=commit [0-9a-f]{40}|\Z)'
    commits = re.findall(pattern, retrieved_content)
    return commits


def extract_commit_id(commit_content: str):
    """
    Extract the commit SHA from a commit block.

    Args:
        commit_content (str): Text of one commit.

    Returns:
        str: Commit SHA (40 hex characters).
    """
    match = re.search(r'commit ([0-9a-f]{40})', commit_content)
    return match.group(1)


def get_commit_content(commit_id_tuple):
    """
    Retrieve full commit details using `git show`.

    Args:
        commit_id_tuple (tuple): (repository name, commit SHA)

    Returns:
        tuple: (repo name, commit SHA, commit full text)
    """
    reponame, commit_id = commit_id_tuple
    cmd = f"cd ../../Repo/{reponame} && git show {commit_id}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='latin-1')
    return reponame, commit_id, result.stdout


def parallel_process_commits(commit_id_list, reponame, num_workers=4):
    """
    Process multiple commits in parallel to fetch their content.

    Args:
        commit_id_list (List[str]): List of commit SHAs.
        reponame (str): Repository folder name.
        num_workers (int): Number of parallel workers.

    Returns:
        List[tuple]: List of (repo, SHA, content) tuples.
    """
    tasks = [(reponame, cid) for cid in commit_id_list]
    with mp.Pool(processes=num_workers) as pool:
        results = list(pool.imap(get_commit_content, tasks))
    return results


def catch_commit_datetime(commit_message: str):
    """
    Extract commit date and return the year.

    Args:
        commit_message (str): Full commit text from `git show`.

    Returns:
        int: Year of the commit.
    """
    match = re.search(r"Date:\s+(.*)", commit_message)
    if match:
        date_str = match.group(1)
        dt = datetime.strptime(date_str, "%a %b %d %H:%M:%S %Y %z")
        return dt.year
    else:
        # If date not found, return None to indicate missing data
        return None


def main(reponame: str):
    """
    Main entry point: search commits by security keywords and save results.

    Args:
        reponame (str): Name of the repository folder.
    """
    # Determine available CPU cores for parallelism
    num_workers = mp.cpu_count()
    print(f"Using {num_workers} worker processes.")

    # Prepare output directory
    base_path = f"../../automated_data/repo/{reponame}/"
    os.makedirs(base_path, exist_ok=True)
    stage0_path = os.path.join(base_path, 's0')
    os.makedirs(stage0_path, exist_ok=True)
    output_file = os.path.join(stage0_path, 's0_0_keyword_match_commits.pkl')

    # Avoid re-running if results already exist
    if os.path.exists(output_file):
        print("Output file already exists, skipping processing.")
        return

    all_commit_info = []
    # Loop through each keyword and find matching commits
    for keyword in tqdm(SECURITY_RELATED_KEYWORDS, desc="Searching keywords"):
        git_cmd = f"cd ../../Repo/{reponame} && git log --grep='{keyword}' -i"
        output = subprocess.run(git_cmd, shell=True, capture_output=True, text=True, encoding='latin-1').stdout
        commits = split_commit_content(output)
        ids = [extract_commit_id(c) for c in commits]
        commit_data = parallel_process_commits(ids, reponame, num_workers)
        all_commit_info.extend(commit_data)

    # Filter out any empty results and deduplicate
    all_commit_info = [(r, cid, txt) for r, cid, txt in all_commit_info if txt]
    unique_info = list(set(all_commit_info))

    # Save results for downstream processing
    with open(output_file, 'wb') as f:
        pickle.dump(unique_info, f)
    print(f"Saved {len(unique_info)} commits to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter commit list by security keywords')
    parser.add_argument('--reponame', type=str, required=True, help='Name of the target repository')
    args = parser.parse_args()
    main(args.reponame)
