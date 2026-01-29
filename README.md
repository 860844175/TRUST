# To Trust, or Not to Trust, That is the Critical Question: Responsible Code Generation with Adversarial Vulnerability Awareness and Calibration Resonance
<img width="488" height="575" alt="Chat with Context TRUST S P" src="https://github.com/user-attachments/assets/64dd71f1-89e4-4596-b0ef-0b71bb715540" />

# Abstract

Large Language Models (LLMs) have demonstrated remarkable capabilities in code generation, yet their outputs often lack trustworthiness, as ambiguity and latent vulnerabilities pervade. When such code is directly integrated into systems, particularly within security-sensitive or safety-critical domains, these vulnerabilities may introduce unforeseen risks and attack surfaces. This paper presents a novel framework, TRUST, for responsible code generation that tackles inherent trust deficits in existing LLM-based code generation systems. Central to our approach is a vulnerability-aware model that operates within an adversarial refinement framework, iteratively improving the security posture of generated code. At each step, the system not only enhances code robustness but also produces interpretable explanations for identified vulnerabilities. To mitigate the risk of model overconfidence, a well-documented issue in generative systems, we introduce a calibration mechanism that enables the LLM to recognize the boundaries of its own trustworthiness. When prompted with inputs beyond its trust boundaries, the model abstains from generation and issues a refusal response, such as ``\textit{I cannot do that... because...}'', thereby preventing the emission of potentially insecure or misleading code. To enable these capabilities, we employ a novel training paradigm that leverages large-scale code version histories and commit logs. This paradigm equips the model with a deeper understanding of code evolution, vulnerability patterns, and trust boundaries, facilitating more secure and context-aware code generation. Evaluation results, based on a newly introduced benchmark dataset \textit{TRUST-Bench}, show that our method establishes a robust and responsible pathway for adversarial code generation with calibration resonance, outperforming state-of-the-art pre-trained LLMs as well as the SafeCoder training framework, which was derived from its predecessor SVEN.


<img width="2164" height="1085" alt="Responsbile AI (4) (1)-1" src="https://github.com/user-attachments/assets/ed0e9e5c-7fe3-40d9-864a-2b4dfe234282" />

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

---

# Appendix

## Detailed TRUST Dataset Construction

To support the training and evaluation of secure code generation models, we construct **TRUST-Bench**, a dataset derived from real-world version control histories of widely used C-based open-source projects. Our goal is to extract high-confidence vulnerability-fix instances along with benign examples to support security-related tasks. This section outlines the data acquisition pipeline, which terminates at the commit level, prior to sample-level processing.

**Data Collection:** For the training dataset, we begin by selecting several widely used C-based open-source repositories (e.g., Android, FFmpeg, OpenSSL), focusing on commits to those before the end of 2022 for the training set. Our data construction follows a multi-stage pipeline designed to extract genuine vulnerability-fix instances while filtering out irrelevant or excessively convoluted commits:

*   **Keyword-based Mining.** We compile a list of security-related keywords (e.g., "fix", "vulnerability", "risk") based on domain knowledge, and flag commits whose messages contain at least one such term. This step yields over *500K* potentially security-relevant commits.

*   **Rule-based Pre-filtering.** Commits that involve non-C files, modify an excessive number of files or code blocks, or result in overly long diffs are discarded to ensure consistency and model compatibility (e.g., avoiding token limit issues during inference).

*   **Semantic Validation.** For the remaining candidates, we use the *GPT-4o* model to assess the security relevance of each commit. A commit is retained if it satisfies both of the following conditions: (1) the pre-fix code contains a potentially vulnerable construct or insecure pattern; and (2) the post-fix change plausibly mitigates that issue. Commits meeting both criteria are considered high-confidence security fixes and are kept for downstream use.

*   **Function-Level Context Extraction and Atomicity Enforcement.** To improve semantic clarity and reduce label ambiguity, we retain only commits that modify a single file and contain a single diff hunk, assuming each reflects a single vulnerability fix. Multi-file or multi-hunk commits often correspond to compound patches or broad hardening efforts involving cross-file dependencies, which are harder to interpret and less suitable for learning localized fix patterns. By focusing on atomic, self-contained patches, we enable more reliable downstream tasks such as vulnerability localization, causal analysis, and patch generation.

Table `Appendix:commit_filtering` summarizes the remaining vulnerability commits after each filtering step.

> [!NOTE]
> **Table Placeholder**: This section references `Table/8_Appendix_Application_Table`. Please add the table content here.

For the test dataset, we observe that pre-trained LLMs, such as *Qwen2.5-Coder* and *DeepSeek-Coder-V2*, have been developed within the past two years. To mitigate the risk of data leakage, where parts of test set may overlap with LLM training data, we exclusively select vulnerabilities published in 2023 and 2024 with assigned CVE identifiers. Additionally, we considered two key factors when curating the test set. First, patches with concentrated code changes are more suitable for masking, as the core modifications can be effectively targeted. Second, if the function containing the masked changes is excessively long, it pose challenges due to the token limitations of LLMs. Balancing these considerations, we filtered the collected vulnerability patches and finalized a test set comprising 274 high-quality instances.

Training solely on vulnerable code is insufficient, which will hinder the ability of inspection models to identify benign cases accurately, compromising their capability to determine the absence of vulnerabilities. To address this, we curated *5,000* benign examples from five repositories (gpac, ImageMagick, vim, openssl, FFmpeg) for balance and representativeness. To ensure the correctness of these benign functions, we leveraged the git log to rank functions by their lifespan within each repository, prioritizing the longest-surviving code as the safest candidates.

**Data Processing:** Providing adequate context information is essential for fairness in code completion tasks with masked code snippets. If the mask references external functions, globally defined variables, or data structures, it is unrealistic to expect LLMs to generate accurate and meaningful completions without access to this contextual information. This limitation is particularly pronounced when the model encounters repositories it has not been exposed to during training. To address this, necessary context information must be included with the masked code snippets during both training and evaluation phases. As our experiment handles function-level code snippets, the relevant context for a given function is primarily composed of its callee functions and any undefined variables referenced within its body. In contrast, the caller function does not directly contribute to the semantic or syntactic structure of the masked function and is therefore excluded from the context.

For the testing data, we ensure data quality by manually collecting all relevant contexts for the masked code. For training data, we adopt a suite of automated definition retrieval techniques. Specifically, for macro definitions, context information is extracted during the precompilation phase through macro expansion, utilizing tools such as Bear. For other definitions, we employ srcML to generate an XML-based representation of each repository. By parsing and querying this structured format, relevant context definitions can be efficiently identified. It is worth noting that a single line of code may involve extensive contextual dependencies, such as functions and variables. However, due to token limitations, we limit the scope of context retrieval to avoid iterative tracing.

The extracted dataset serves as the foundation for training the two key modules in our system pipeline.

Each collected commit provides a structured triplet:
$(C_{pre}, m, C_{fix})$, where $C_{pre}$ denotes the pre-fix (vulnerable) code, $C_{fix}$ is the corresponding post-fix (patched) code, and $m$ is the commit message describing the change. These triplets serve as supervision for two learning objectives:

*   **Trust Inspection (Task 1):** The model learns to identify vulnerable code spans in $C_{pre}$ and generate natural language explanations that align with the semantics of the corresponding patch $(C_{fix})$ and message $m$.
*   **Calibration (Task 2):** By comparing LLM-generated patches to the gold-standard post-fix code $C_{fix}$, the model learns to assess whether a proposed refinement successfully mitigates the identified vulnerabilities.

## Iterative Refinement Stopping Criteria

To prevent unnecessary computation and avoid infinite refinement loops, we define the termination of iterative code refinement based on a hybrid strategy incorporating three signals as described in Section `adversarial`. Below, we provide implementation-level details for each criterion:

**1. No Vulnerability Detected**: If the Trust Inspector returns "No vulnerability Detected", indicating that no further issues are identified in the current generation, the loop is terminated.

**2. High Token-Level Similarity**: We compute the normalized token-level overlap between two successive code generations $C^{(t)}_{\text{gen}}$ and $C^{(t+1)}_{\text{gen}}$. Tokens are obtained using the LLM’s native tokenizer (matching the model used in generation). Let $T^{(t)}$ and $T^{(t+1)}$ denote the token sequences at iteration $t$ and $t+1$, respectively. We define similarity as:

$$
\text{Sim}(T^{(t)}, T^{(t+1)}) = \frac{|T^{(t)} \cap T^{(t+1)}|}{|T^{(t)} \cup T^{(t+1)}|}
$$

If this token-level Jaccard similarity exceeds a threshold $\tau$, the refinement is considered converged. In our experiments, we set $\tau = 0.90$, which balances tolerance for minor edits with sensitivity to substantive changes. This threshold was inspired by convergence heuristics used in code mutation and prompt-editing tasks in prior work.

**3. Maximum Iteration Limit**: To maintain computational efficiency and ensure bounded runtime, we impose a fixed upper bound on the number of iterations. In our implementation, we set this limit to 2 refinement rounds per input. We found this conservative limit sufficient for the majority of test cases, where either no vulnerabilities were detected after one pass or the second iteration resolved remaining issues.

## Sample Distribution

As discussed in Section `4.3_RQ2_Adversarial`, our dataset covers 39 distinct CWE categories. By traversing the CWE-ID hierarchy, we are able to identify the top-level (ancestor) category for each specific CWE-ID. Out of the 274 total cases in our dataset, 128 can be mapped to the official CWE Top 25 list. The remaining cases are associated with CVE identifiers but lack a corresponding CWE mapping. Figure below illustrates the distribution of these top-level CWEs within our dataset.

![Distribution of Non-Null CWE Samples in the Test Set Mapped to CWE Top-25.](docs/appendix/assets/CWE-distribution.png)
*Figure: Distribution of Non-Null CWE Samples in the Test Set Mapped to CWE Top-25.*

> [!NOTE]
> Please ensure `CWE-distribution.png` is placed in the `docs/appendix/assets/` directory.

## Code length and model uncertainty

![The relationship between code snippet length and model uncertainty.](docs/appendix/assets/Appendix_PPL_vs_TokenLength.png)
*Figure: The relationship between code snippet length and model uncertainty.*

> [!NOTE]
> Please ensure `Appendix_PPL_vs_TokenLength.png` is placed in the `docs/appendix/assets/` directory.

Perplexity (PPL) is a metric commonly used to measure how confident a language model is when making predictions. In simple terms, it reflects how "confused" the model is when processing a given input. A lower PPL means the model finds the input more predictable or familiar, while a higher PPL indicates that the model is uncertain or surprised by the input.

In the context of code understanding, if a model assigns high perplexity to a code snippet, it suggests the model is unsure how to interpret or complete that code, possibly due to unfamiliar patterns, poor structure, or lack of training exposure. Conversely, low perplexity implies the model can follow the logic more easily.

The figure above illustrates a clear inverse relationship: code snippets with low perplexity (`PPL<1000`) tend to be longer, suggesting that longer, well-structured code can provide better context for the model to understand. On the other hand, code snippets with high perplexity (`PPL>200000`) are typically short and fragmented, making them harder for the model to interpret confidently.

## Human Evaluation Rubric for TRUST-Bench

> [!NOTE]
> **Table Placeholder**: This section references `Table/8_Appendix_Evaluation_Rubric`. Please add the table content here.

To ensure the quality and reliability of TRUST-Bench, we conduct a structured human evaluation based on the rubric in the table referenced above.
The rubric evaluates five aspects critical for vulnerability-aware supervision: vulnerability mask accuracy, explanation–code alignment, explanation completeness, semantic consistency, and overall sample validity.
Each criterion is assessed using binary decisions (Pass/Fail) to minimize subjectivity, while overall validity determines whether a sample is retained, revised, or removed.

The evaluation is independently performed by three security experts.
Expert E1 holds a Ph.D. in computer security with over three years of experience in vulnerability analysis and secure software development.
Expert E2 is a senior security engineer with extensive industry experience in auditing real-world vulnerabilities and analyzing security patches.
Expert E3 is a doctoral researcher specializing in program analysis and vulnerability detection, with prior experience in constructing security benchmarks.
All experts annotate independently without access to each other’s results.

Inter-annotator agreement is measured using Fleiss’ Kappa. The overall agreement reaches $\kappa = 0.68$, indicating substantial agreement, with per-criterion Kappa values ranging from $0.56$ to $0.77$. Higher agreement is observed for vulnerability mask accuracy and semantic consistency. Disagreements are resolved through majority voting, and samples failing multiple criteria are revised or removed, ensuring reliable benchmark construction.

## Inference Overhead Analysis

> [!NOTE]
> **Table Placeholder**: This section references `Table/8_latency`. Please add the table content here.

> [!NOTE]
> **Table Placeholder**: This section references `Table/8_token`. Please add the table content here.

This appendix provides a detailed breakdown of the inference overhead introduced by TRUST, including both end-to-end latency and token consumption.
We report results under two configurations: the original API-based generation pipeline and the TRUST-enabled pipeline.
All results are averaged over the evaluation samples, and we report the mean and standard deviation.

The latency table presents a stage-wise latency breakdown across the full pipeline, including code generation, vulnerability localization, explanation, iterative refinement, and calibration.
Although TRUST introduces additional security-oriented stages, the overall end-to-end latency consistently decreases compared to the corresponding API baselines.
This reduction is primarily attributed to shorter and more stable refinement stages, as early vulnerability inspection and calibrated control reduce redundant iterations and prevent unstable generation trajectories.

The token table reports the corresponding token consumption breakdown.
Across models, TRUST consistently reduces total token usage, with notable savings in the explanation and refinement stages.
These reductions indicate that early inspection and calibration help suppress unnecessary verbose explanations and repeated refinements, leading to more efficient generation.
Overall, the results demonstrate that TRUST not only improves the security of generated code, but also reduces computational overhead, making it practical for deployment in real-world software development workflows.

## Prompt Templates

> [!NOTE]
> **Table Placeholder**: This section references `Table/8_Appendix_Prompt_Design`. Please add the table content here.

Table above presents the carefully designed prompts used throughout both the fine-tuning and evaluation stages of the TRUST pipeline. Each prompt corresponds to a distinct task in our multi-step vulnerability assessment framework, including vulnerability identification, vulnerability explanation, code refinement, and confidence calibration.
