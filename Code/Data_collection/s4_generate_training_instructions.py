import os
import json
import pickle
import argparse
import random

# Load our instruction templates (system + user prompts) from JSON
instruction_templates = json.load(open('./instruction_template.json', 'r'))

def main(reponame: str):
    """
    Stage s4: Assemble LLM fine‑tuning examples for vulnerability localization & explanation.

    1. Load the short commits and contexts from s3.
    2. Load the s4 locate & explain labels.
    3. For each example, build two training records:
       - One for the "locate" task.
       - One for the "locate+explain" task.
    4. Shuffle and dump as JSON for downstream fine‑tuning.
    """
    # Paths to s3/s4 data
    s3_dir = f'./data/repo/{reponame}/s3/'
    s4_dir = f'./data/repo/{reponame}/s4/'
    # Load s3 data
    commits      = pickle.load(open(os.path.join(s3_dir, 's3_1_short_commit_list.pkl'),       'rb'))
    contexts     = pickle.load(open(os.path.join(s3_dir, 's3_1_short_context_str_list.pkl'),  'rb'))
    # Load s4 labels (v2 versions)
    locate_labels       = pickle.load(open(os.path.join(s4_dir, 's4_locate_label_v2.pkl'),           'rb'))
    explain_labels      = pickle.load(open(os.path.join(s4_dir, 's4_locate_explain_label_v2.pkl'), 'rb'))

    # Sanity check
    assert len(commits) == len(contexts) == len(locate_labels) == len(explain_labels)
    print(f"Preparing {len(commits)} instruction pairs for {reponame}")

    train_examples = []

    for idx, entry in enumerate(commits):
        # unpack: entry[2] is full diff, entry[-6] prefix, entry[-5] fix
        _, _, commit_diff, *_, prefix_code, fix_code = entry
        context_str   = contexts[idx]
        loc_label     = locate_labels[idx]
        expl_label    = explain_labels[idx]

        # 1) "locate" task example
        train_examples.append({
            "instruction": instruction_templates['locate']['system_prompt'],
            "input":       instruction_templates['locate']['user_message'].format(prefix_code, context_str),
            "output":      loc_label
        })

        # 2) "locate + explain" task example
        train_examples.append({
            "instruction": instruction_templates['locate_explain']['system_prompt'],
            "input":       instruction_templates['locate_explain']['user_message']
                                .format(prefix_code, loc_label, context_str),
            "output":      expl_label
        })

    # Shuffle for randomness
    random.shuffle(train_examples)

    # Normalize repo name to filesystem-friendly
    safe_name = reponame.replace('/', '-')
    out_path = os.path.join(s4_dir, f'{safe_name}_train_instruction_list.json')

    # Dump to JSON
    with open(out_path, 'w') as f:
        json.dump(train_examples, f, indent=4)

    print(f"Saved {len(train_examples)} examples to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate fine‑tuning instructions from s4 labels')
    parser.add_argument('--reponame', type=str, required=True,
                        help='Repository name (e.g. FFmpeg-FFmpeg or Android/kernel-common)')
    args = parser.parse_args()
    main(args.reponame)
