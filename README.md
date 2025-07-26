# To Trust, or Not to Trust, That is the Critical Question: Responsible Code Generation with Adversarial Vulnerability Awareness and Calibration Resonance

# Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, yet their outputs often lack trustworthiness, as ambiguity and latent vulnerabilities pervade. When such code is directly integrated into systems, particularly within security-sensitive or safety-critical domains, these vulnerabilities may introduce unforeseen risks and attack surfaces. This paper presents a novel framework, TRUST, for responsible code generation that tackles inherent trust deficits in existing LLM-based code generation systems. Central to our approach is a vulnerability-aware model that operates within an adversarial refinement framework, iteratively improving the security posture of generated code. At each step, the system not only enhances code robustness but also produces interpretable explanations for identified vulnerabilities. To mitigate the risk of model overconfidence, a well-documented issue in generative systems, we introduce a calibration mechanism that enables the LLM to recognize the boundaries of its own trustworthiness. When prompted with inputs beyond its trust boundaries, the model abstains from generation and issues a refusal response, such as ``\textit{I cannot do that... because...}'', thereby preventing the emission of potentially insecure or misleading code. To enable these capabilities, we employ a novel training paradigm that leverages large-scale code version histories and commit logs. This paradigm equips the model with a deeper understanding of code evolution, vulnerability patterns, and trust boundaries, facilitating more secure and context-aware code generation. Evaluation results, based on a newly introduced benchmark dataset \textit{TRUST-Bench}, show that our method establishes a robust and responsible pathway for adversarial code generation with calibration resonance, outperforming state-of-the-art pre-trained LLMs as well as the SafeCoder training framework, which was derived from its predecessor SVEN.

# Data Collection

This directory contains the** ****data‑preparation** stages. It walks raw Git histories through a multi‑step workflow to produce high‑quality, vulnerability‑focused training data.

## Stage **s0** – Commit Retrieval & Initial Filtering

1. **`s0_1_keyword_match_commits.py`**

   - **Scan** each repository for security‑related keywords in commit messages.
   - **Fetch** full diffs for all matching commits via `git log` + `git show`.
   - **Output:** `s0/s0_0_keyword_match_commits.pkl`
2. **`s0_2_filter_security_commits.py`**

   - **Drop** commits with enormous diffs (> 1 million chars).
   - **Filter** by token length (< 1000 tokens), single‑file changes, and C/C++ extensions.
   - **Output:** `s0/s0_2_security_commits_filtered.pkl`
3. **`s0_3_commit_security_analysis.py`**

   - **Invoke** an LLM (Meta‑Llama) to **classify** each diff as a real vulnerability fix (`yes`/`no`/`cannot decide`).
   - **Batch** processing with `--start`/`--end` options for large datasets.
   - **Output:** `s0/s0_3_commit_security_analysis_results.pkl`

---



## Stage **s1** – Function‑Level Refinement & Mask Generation

1. **`s1_1_refine_security_commits.py`**

   - **Select** only those commits labeled `yes` by the LLM.
   - **Enrich** each commit with full “before” & “after” function bodies.
   - **Remove** no‑ops, multi‑hunk patches, and commits with only formatting changes.
   - **Filter** by function length (< 1000 tokens), deletion presence, and date (< 2023).
   - **Output:** `s1/s1_1_security_commits_updated.pkl`
2. **`s1_2_mask_vulnerability_blocks.py`**

   - **Prompt** an LLM to detect the exact risky code block within each function.
   - **Replace** that block with a `<MASK_n>` token, preserving surrounding context.
   - **Output:** `s1/s1_2_security_commits_blank_results.pkl`

---

## Stage **s2** – Mask Reintegration & Context Extraction

1. **`s2_0_finalize_security_commits.py`**

   - **Merge** `<MASK>` annotations back into full commit records.
   - **Extract** “undefined” code elements adjacent to each mask.
   - **Output:** `s2/s2_0_security_commits_single_mask_with_undefined_elements.pkl`
2. **`s2_1_analyze_undefined_elements.py`**

   - **Task 1:** Extract **all** code elements (functions, structs, variables) mentioned in masked snippets.
   - **Task 2:** Locate each element’s **definition** or **assignment** in the full code.
   - **Output:**
     - `s2/s2_1_task1_outputs.pkl`
     - `s2/s2_1_task2_outputs.pkl`
3. **`s2_2_validate_and_filter_undefined_elements.py`**

   - **Parse** raw LLM outputs into structured Python lists/dicts.
   - **Filter** out common library calls, error codes, and overly short names.
   - **Output:**
     - `s2/s2_2_valid_task1_output_list.pkl`
     - `s2/s2_2_valid_task2_output_list.pkl`
     - `s2/s2_2_valid_s2_0_commits.pkl`

---


## Stage **s3** – Vulnerability Localization & Explanation Label Generation

1. **`s3_1_locate_vulnerable_segments.py`**

   - **Prompt** an LLM to pinpoint the exact vulnerable code blocks within each prefix function.
   - **Output:** `s4/s4_locate_vulnerable_segments.pkl`
2. **`s3_2_explain_vulnerable_segments.py`**

   - **Explain** why each located code segment is vulnerable, detailing root cause and impact.
   - **Output:** `s4/s4_explain_vulnerable_segments.pkl`

---


## Stage s4 - Instruction Generation

1. `s4_generation_training_instruction.py`
