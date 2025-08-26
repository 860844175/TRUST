import pickle
import os
import argparse
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
from s2_utils import *

# Number of parallel workers
number_of_workers = mp.cpu_count()

def main(reponame):
    """
    Stage s2_0:  
      1. Load masked results (s1_2) and the original filtered commits (s1_1).  
      2. Drop any empty mask outputs and their corresponding commits.  
      3. Apply the <MASK> tokens back into the full commit entries.  
      4. Keep only those commits with exactly one mask.  
      5. For each single-mask commit, extract any undefined elements around the mask.  
      6. Discard entries with no undefined elements and save the rest.
    
    Args:
        reponame (str): Name of the repository folder (e.g. "FFmpeg-FFmpeg").
    """
    # Paths for input data
    path_s1_1 = f'../../automated_data/repo/{reponame}/s1/s1_1_security_commits_updated.pkl'
    path_s1_2 = f'../../automated_data/repo/{reponame}/s1/s1_2_security_commits_blank_results.pkl'

    # 1. Load stage‑1 data
    commits_updated = pickle.load(open(path_s1_1, 'rb'))
    blank_results    = pickle.load(open(path_s1_2, 'rb'))

    # 2. Remove any empty mask outputs
    non_empty_masks = [res for res in blank_results if res]
    # Keep only the commits whose mask result was non-empty
    commits_non_empty = [
        commit for commit, mask in zip(commits_updated, blank_results) if mask
    ]

    # 3. Insert the <MASK> tokens back into each commit entry
    #    add_mask_to_list returns (list_with_masks, indices_of_bad_entries)
    masked_commits, bad_indices = add_mask_to_list(commits_non_empty, non_empty_masks)

    # Remove entries flagged as bad
    commits_clean = [
        c for idx, c in enumerate(commits_non_empty) if idx not in bad_indices
    ]
    masks_clean   = [
        m for idx, m in enumerate(non_empty_masks) if idx not in bad_indices
    ]

    # Ensure alignment after filtering
    assert len(commits_clean) == len(masks_clean) == len(masked_commits)

    # 4. Keep only those with exactly one <MASK_*> token
    single_mask_commits = [
        entry for entry in masked_commits
        if entry[-1].count("<MASK_") == 1
    ]

    # 5. For each single‑mask commit, extract undefined elements in parallel
    with Pool(number_of_workers) as pool:
        enriched = list(tqdm(
            pool.imap(get_undefined_element_to_list, single_mask_commits),
            total=len(single_mask_commits),
            desc="Extracting undefined elements"
        ))

    # 6. Drop any entries where no undefined element was found
    final_entries = [
        e for e in enriched
        if e[-2] or e[-3]  # either prefix or fix undefined list is non-empty
    ]

    # Save the final list for s2 stage
    out_path = f'../../automated_data/repo/{reponame}/s2/s2_0_security_commits_single_mask_with_undefined_elements.pkl'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(final_entries, f)

    print(f"Saved {len(final_entries)} entries to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage s2_0: finalize masked security commits')
    parser.add_argument(
        '--reponame', type=str, required=True,
        help='Repository name, e.g. FFmpeg-FFmpeg or Android/kernel-common'
    )
    args = parser.parse_args()

    # Ensure the s2 folder exists
    s2_folder = f'../../automated_data/repo/{args.reponame}/s2'
    os.makedirs(s2_folder, exist_ok=True)

    main(args.reponame)
