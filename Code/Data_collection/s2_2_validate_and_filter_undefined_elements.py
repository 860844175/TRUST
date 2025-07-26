import os
import pickle
import argparse
from tqdm import tqdm
from s2_utils import transform_str_to_list

"""
Stage s2_2: Validate and filter the undefined‐element extraction results.

This script takes:
  - s2_0 entries (single‐mask commits with undefined elements)
  - s2_1 Task 1 and Task 2 outputs

It then:
  1. Drops any Task 2 outputs that are empty or not parseable.
  2. Keeps only the corresponding s2_0 commits and Task 1 outputs.
  3. Converts the valid Task 2 strings into Python lists/dicts.
  4. Filters out common/uninteresting items (e.g. standard library functions, error codes).
  5. Saves three filtered pickle files for downstream processing.
"""

# A built‐in list of too‐common items to exclude in final filtering
COMMON_BLACKLIST = [
    'strcmp', 'strncpy', 'strcpy', 'strlen', 'memset', 'memcpy',
    'fprintf', 'exit', 'malloc', 'free', 'sizeof', 'getenv', 'stat',
    'perror', 'closesocket', 'htons', 'sscanf', 'strrchr', 'strchr',
    'tolower', 'toupper',
    'char', 'unsigned char', 'size_t', 'uint16_t', 'uint8_t', 'NULL',
    'ENOMEM', 'EIO', 'EINVAL', 'EAGAIN', 'EPERM', 'ENOENT', 'EEXIST',
    'EPIPE', 'ERANGE', 'EACCES', 'EFAULT', 'EBUSY', 'ENODEV', 'EOVERFLOW',
    'ETIMEDOUT', 'EINTR', 'ECANCELED', 'EADDRINUSE', 'ENOTSUP', 'EISDIR',
    'ENOTDIR', 'ENOTEMPTY'
]

def filter_list_of_dicts(data):
    """
    Filter each dict in a list by removing blacklisted or too‐short items.

    Args:
        data (List[dict]): Each dict has keys 'Functions', 'Variables', 'Structures'.

    Returns:
        List[dict]: Same structure, but with unwanted names removed.
    """
    def filter_elements(elements):
        # Only keep items longer than 2 characters and not in our common blacklist
        return [item for item in elements
                if len(item) > 2 and item not in COMMON_BLACKLIST]

    filtered = []
    for entry in data:
        new_entry = {}
        for k, v in entry.items():
            if isinstance(v, list):
                new_entry[k] = filter_elements(v)
            else:
                new_entry[k] = v
        filtered.append(new_entry)
    return filtered

def main(reponame):
    """
    Main procedure for s2_2.

    1. Load s2_0 commits and s2_1 Task 1/Task 2 outputs.
    2. Identify which Task 2 outputs are valid (wrapped in backticks and parseable).
    3. Keep only the matching s2_0 commits and Task 1 outputs.
    4. Transform valid Task 2 strings into Python dicts.
    5. Filter out common/uninteresting elements.
    6. Save:
       - valid Task 2 lists
       - valid Task 1 strings
       - corresponding s2_0 commit entries
    """
    base = f'../../automated_data/repo/{reponame}/s2'
    # Paths to load
    path_s2_0   = os.path.join(base, 's2_0_security_commits_single_mask_with_undefined_elements.pkl')
    path_t1_out = os.path.join(base, 's2_1_task1_outputs.pkl')
    path_t2_out = os.path.join(base, 's2_1_task2_outputs.pkl')

    # 1. Load data
    commits0 = pickle.load(open(path_s2_0, 'rb'))
    task1   = pickle.load(open(path_t1_out, 'rb'))
    task2   = pickle.load(open(path_t2_out, 'rb'))

    # 2. Find indices of Task 2 outputs that are non-empty and parseable
    valid_indices = []
    for idx, txt in enumerate(task2):
        if not txt:
            continue  # skip empty
        # Expect the LLM to have wrapped the JSON‐like list in ```…``` 
        parts = txt.split('```')
        if len(parts) < 3:
            continue
        candidate = parts[-2]
        try:
            _ = transform_str_to_list(candidate)
            valid_indices.append(idx)
        except Exception:
            continue

    # 3. Keep only the valid entries from each source
    commits_valid = [commits0[i] for i in valid_indices]
    task1_valid   = [task1[i]   for i in valid_indices]
    task2_raw     = [task2[i].split('```')[-2] for i in valid_indices]

    # 4. Convert Task 2 raw strings into Python lists/dicts
    task2_list = [transform_str_to_list(s) for s in task2_raw]

    # 5. Filter out common/uninteresting names
    task2_filtered = filter_list_of_dicts(task2_list)

    # 6. Save filtered results
    pickle.dump(task2_filtered, open(os.path.join(base, 's2_2_valid_task2_output_list.pkl'), 'wb'))
    pickle.dump(task1_valid,   open(os.path.join(base, 's2_2_valid_task1_output_list.pkl'),   'wb'))
    pickle.dump(commits_valid, open(os.path.join(base, 's2_2_valid_s2_0_commits.pkl'),      'wb'))

    print(f"Saved {len(task2_filtered)} valid Task 2 entries, "
          f"{len(task1_valid)} Task 1 entries, and {len(commits_valid)} commits.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage s2_2: validate & filter undefined elements')
    parser.add_argument('--reponame', type=str, required=True, help='Repository name')
    args = parser.parse_args()
    main(args.reponame)
