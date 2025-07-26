import os
import pickle
import argparse
from vllm import LLM, SamplingParams

prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}

<|eot_id|><|start_header_id|>user<|end_header_id|>
{user}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

system_prompt = """
You are a software security expert. Your task is to analyze commit information and code diffs to pinpoint the vulnerable segments in the prefix code. Focus only on the vulnerabilities the commit intends to fix, using the commit message, prefix code, and supplied context.

Key requirements:
1. **Assume fix code is unavailable**—infer vulnerability purely from prefix code and commit description.
2. **Sensitive operations**: Pay attention to memory handling, input processing, file/network operations.
3. **Data flow**: Trace untrusted data paths to potential misuse.
4. **Control points**: Highlight conditionals, loops, or branches that may be insecure.

**Line Number Rules**:
- Line numbers must refer **only to the prefix code** provided.
- Do **not** use diff-based or commit-based line numbers.
- If you cannot determine line numbers, use `[unknown]`.

**Response Format**:
```

1. Vulnerable Code Blocks and Lines:

* **Code Block 1**:

  * Function Name: <name or [unknown]>
  * Code Snippet:

    ```
    <exact lines from prefix code>
    ```
  * Line Numbers: [start-end] or [unknown]
    ...

```
"""

user_message_template = """
I will provide:
1) Commit information (message + diff)
2) Prefix code (complete function before the fix)
3) Fix code (complete function after the fix)
4) Context List (definitions of all functions referenced)

Inputs:

1) Commit Information:
```

{commit_info}

```

2) Prefix Code:
```

{prefix_code}

```

3) Fix Code:
```

{fix_code}

```

4) Context List:
```

{context_list}

```
"""

def main(reponame: str):
    """
    Stage s4: Locate vulnerable segments in prefix code.

    Loads the stage‑3 filtered commits and contexts, builds detailed LLM prompts,
    invokes the model to extract vulnerable code blocks with accurate line numbers,
    and saves the results for downstream processing.

    Args:
        reponame: Repository identifier, e.g. "FFmpeg-FFmpeg".
    """
    # Paths for input/output
    s3_path = f'./data/repo/{reponame}/s3'
    s4_path = f'./data/repo/{reponame}/s4'
    os.makedirs(s4_path, exist_ok=True)

    commits = pickle.load(open(os.path.join(s3_path, 's3_1_short_commit_list.pkl'), 'rb'))
    contexts = pickle.load(open(os.path.join(s3_path, 's3_1_short_context_str_list.pkl'), 'rb'))

    assert len(commits) == len(contexts), "Commit and context counts must match"

    # Prepare LLM
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    llm = LLM(
        model=model_id,
        tensor_parallel_size=4,
        disable_custom_all_reduce=True,
        download_dir="/home/jyu7/"
    )
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=10240)

    # Build prompts
    prompts = []
    for (repo, sha, diff_text, *_, prefix_code, fix_code), context in zip(commits, contexts):
        user_message = user_message_template.format(
            commit_info=diff_text.strip(),
            prefix_code=prefix_code.strip(),
            fix_code=fix_code.strip(),
            context_list=context.strip()
        )
        prompts.append(prompt_template.format(system=system_prompt, user=user_message))

    # Invoke LLM in batches
    outputs = llm.generate(prompts, sampling_params)
    results = [out.outputs[0].text.strip() for out in outputs]

    # Save outputs
    output_file = os.path.join(s4_path, 's4_locate_vulnerable_segments.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Saved {len(results)} vulnerability-location outputs to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage s3_2: Locate vulnerable code segments')
    parser.add_argument('--reponame', type=str, required=True, help='Repository name, e.g. FFmpeg-FFmpeg')
    args = parser.parse_args()
    main(args.reponame)
