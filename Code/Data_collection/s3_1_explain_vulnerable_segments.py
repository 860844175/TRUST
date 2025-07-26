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
You are a software security expert. Your task is to explain why identified code segments are vulnerable, based solely on their prefix code context.  

### Analysis Guidelines  
1. **Independent Analysis**  
   - Focus on inherent security weaknesses in the provided code segment.  
   - Do not reference fixes or commit messages.  

2. **Technical Depth**  
   - State the root cause of the vulnerability in precise terms.  
   - Describe potential security impacts if exploited.  

3. **Direct Style**  
   - Omit introductory phrases; present analysis as concise conclusions.  

### Response Format  

1) Code Segment Details:  
   - Function Name: `<name or [unknown]>`  
   - Code Snippet:  
     ```  
     <exact lines from prefix code>  
     ```  

2) Explanation:  
   - Root Cause: `<...>`  
   - Impact: `<...>`  
"""

user_message_template = """
I will provide you with:  
1) Commit information (message + diff)  
2) Identified vulnerable code segments  
3) Prefix code (before fix)  
4) Fix code (after fix)  
5) Context list (definitions from prefix & fix)  

Inputs:

1) Commit Information:
````

{commit_info}

```

2) Vulnerable Code Segments:
```

{vuln_segments}

```

3) Prefix Code:
```

{prefix_code}

```

4) Fix Code:
```

{fix_code}

```

5) Context List:
```

{context_list}

```
"""

def main(reponame: str):
    """
    Stage s4.2: Explain the vulnerabilities in located code segments.

    1. Load stage‑3 commits and contexts.
    2. Load stage‑4 located segments.
    3. Build and send LLM prompts to explain each segment’s root cause & impact.
    4. Save the textual explanations.
    """
    # Paths
    s3_dir = f'./data/repo/{reponame}/s3'
    s4_dir = f'./data/repo/{reponame}/s4'
    os.makedirs(s4_dir, exist_ok=True)

    # 1. Load filtered commits and contexts
    commits = pickle.load(open(os.path.join(s3_dir, 's3_1_short_commit_list.pkl'), 'rb'))
    contexts = pickle.load(open(os.path.join(s3_dir, 's3_1_short_context_str_list.pkl'), 'rb'))
    assert len(commits) == len(contexts)

    # 2. Load previously located vulnerable segments
    located = pickle.load(open(os.path.join(s4_dir, 's4_locate_vulnerable_segments.pkl'), 'rb'))
    assert len(located) == len(commits)

    # 3. Prepare LLM
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    llm = LLM(
        model=model_id,
        tensor_parallel_size=4,
        disable_custom_all_reduce=True,
        download_dir="/home/jyu7/"
    )
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=8096)

    # 4. Build prompts
    prompts = []
    for (repo, sha, diff_text, *_, prefix_code, fix_code), segments, context in zip(commits, located, contexts):
        user_message = user_message_template.format(
            commit_info=diff_text.strip(),
            vuln_segments=segments.strip(),
            prefix_code=prefix_code.strip(),
            fix_code=fix_code.strip(),
            context_list=context.strip()
        )
        prompts.append(prompt_template.format(system=system_prompt, user=user_message))

    # 5. Generate explanations
    outputs = llm.generate(prompts, sampling_params)
    explanations = [out.outputs[0].text.strip() for out in outputs]

    # 6. Save results
    out_file = os.path.join(s4_dir, 's4_explain_vulnerable_segments.pkl')
    with open(out_file, 'wb') as f:
        pickle.dump(explanations, f)

    print(f"Saved {len(explanations)} vulnerability explanations to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stage s3_1: explain vulnerable segments')
    parser.add_argument('--reponame', type=str, required=True, help='Repository name, e.g. FFmpeg-FFmpeg')
    args = parser.parse_args()
    main(args.reponame)
