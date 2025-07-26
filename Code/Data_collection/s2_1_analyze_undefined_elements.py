import os
import pickle
import argparse
from glob import glob
from tqdm import tqdm
from vllm import LLM, SamplingParams
import tiktoken

"""
Stage s2_1: Identify and classify code elements in masked security commits.

This script runs after retrieving undefined elements (s2_0).  
Input:  
  - A list of tuples, each containing a single‑mask commit entry with undefined elements:
    (repo, sha, diff, changed_file, prefix_fn, fix_fn,
     prefix_ctx, fix_ctx, undefined_prefix_list, undefined_fix_list, mask_code)
Output:  
  - task1_outputs.pkl: for each entry, a deduped list of all code elements (functions, structs, vars, etc.) found in prefix & fix snippets.
  - task2_outputs.pkl: for each entry, classification of those elements into Functions, Variables, and Structures with their definition/assignment lines.
"""

# System prompt for Task 1: extract all elements from pre- and post‑version snippets
task1_system = """
Assume you are a software expert specializing in C code. I will provide you with two code snippets:

1) Pre-version code snippet
2) After-version code snippet

### Task 1: Element Extraction
- Identify all elements (functions, structures, variables, constants, etc.) mentioned in both snippets.
- Exclude duplicates.
- Annotate each element with its type:
  - Variable: regular variable
  - Variable, Pointer: pointer variable (e.g., ptr->member)
  - Variable, Member: struct member variable
  - Struct: struct definition
  - Function: function name

**Output format**:  
[name] (type description), one per line.
"""

task1_user = """Here are the snippets:

Pre-version code snippet:
```

{}

```

After-version code snippet:
```

{}

```"""

# System prompt for Task 2: locate definitions/assignments in full code
task2_system = """
Assume you are a software expert specializing in C code. I will provide you with:

1) A list of elements (functions, structures, variables, constants)
2) The full pre-version code file
3) The full after-version code file

### Task 2: Definition & Origin Location
- Only process elements from input (1).
- Remove all struct member variables from the working set.
- For each remaining element:
  - If it is a regular variable:
    - Find its definition or assignment line in either full code file.
      - If found: output the exact code line and remove it from the variables list.
      - If not found: keep it in the variables list.
  - If it is a pointer variable:
    - Find the line where the pointer is defined; extract the struct type it points to.
    - Add that struct type to the Structures list and remove the pointer variable from Variables.

**Output**:
```

Functions = \[...,]
Variables = \[...,]
Structures = \[...,]

```
"""

task2_user = """Here is the input:

1) List of elements:
```

{}

```
2) Full pre-version code:
```

{}

```
3) Full after-version code:
```

{}

```"""

# Prompt wrapper
prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}

<|eot_id|><|start_header_id|>user<|end_header_id|>
{user}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

def main(reponame):
    """
    Load the single-mask commits with undefined elements, run two LLM tasks, and save outputs.

    Args:
        reponame (str): Repository identifier (e.g. "FFmpeg-FFmpeg").
    """
    base_s2 = f'../../automated_data/repo/{reponame}/s2'
    input_path = os.path.join(base_s2, 's2_0_security_commits_single_mask_with_undefined_elements.pkl')
    commits = pickle.load(open(input_path, 'rb'))

    # Prepare output file paths
    task1_out = os.path.join(base_s2, 's2_1_task1_outputs.pkl')
    task2_out = os.path.join(base_s2, 's2_1_task2_outputs.pkl')

    # Initialize LLM once
    model_id = "meta-llama/Llama-3.1-70B-Instruct"
    llm = LLM(
        model=model_id,
        tensor_parallel_size=4,
        disable_custom_all_reduce=True,
        download_dir="/home/jyu7/"
    )
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=40960)

    # === Task 1: element extraction ===
    # Build prompts: feed each commit's prefix & fix snippets
    task1_prompts = [
        prompt_template.format(
            system=task1_system,
            user=task1_user.format(entry[-3], entry[-2])
        )
        for entry in commits
    ]
    print(f"Running Task 1 on {len(task1_prompts)} entries...")
    task1_results = llm.generate(task1_prompts, sampling_params)
    task1_output = [out.outputs[0].text.strip() for out in task1_results]

    # === Task 2: definition/origin location ===
    # Build prompts: feed Task 1 output plus full code contexts
    task2_prompts = [
        prompt_template.format(
            system=task2_system,
            user=task2_user.format(task1_output[i], entry[-6], entry[-5])
        )
        for i, entry in enumerate(commits)
    ]
    print(f"Running Task 2 on {len(task2_prompts)} entries...")
    task2_results = llm.generate(task2_prompts, sampling_params)
    task2_output = [out.outputs[0].text.strip() for out in task2_results]

    # Save both task outputs
    with open(task1_out, 'wb') as f1:
        pickle.dump(task1_output, f1)
    with open(task2_out, 'wb') as f2:
        pickle.dump(task2_output, f2)

    print(f"Saved Task 1 outputs to {task1_out}")
    print(f"Saved Task 2 outputs to {task2_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage s2_1: analyze undefined code elements')
    parser.add_argument(
        '--reponame', type=str, required=True,
        help='Repository name (e.g. FFmpeg-FFmpeg or Android/kernel-common)'
    )
    args = parser.parse_args()
    main(args.reponame)

