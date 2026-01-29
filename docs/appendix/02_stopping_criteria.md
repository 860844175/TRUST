# Iterative Refinement Stopping Criteria

To prevent unnecessary computation and avoid infinite refinement loops, we define the termination of iterative code refinement based on a hybrid strategy incorporating three signals as described in Section `adversarial`. Below, we provide implementation-level details for each criterion:

**1. No Vulnerability Detected**: If the Trust Inspector returns "No vulnerability Detected", indicating that no further issues are identified in the current generation, the loop is terminated.

**2. High Token-Level Similarity**: We compute the normalized token-level overlap between two successive code generations $C^{(t)}_{\text{gen}}$ and $C^{(t+1)}_{\text{gen}}$. Tokens are obtained using the LLM’s native tokenizer (matching the model used in generation). Let $T^{(t)}$ and $T^{(t+1)}$ denote the token sequences at iteration $t$ and $t+1$, respectively. We define similarity as:

$$
\text{Sim}(T^{(t)}, T^{(t+1)}) = \frac{|T^{(t)} \cap T^{(t+1)}|}{|T^{(t)} \cup T^{(t+1)}|}
$$

If this token-level Jaccard similarity exceeds a threshold $\tau$, the refinement is considered converged. In our experiments, we set $\tau = 0.90$, which balances tolerance for minor edits with sensitivity to substantive changes. This threshold was inspired by convergence heuristics used in code mutation and prompt-editing tasks in prior work.

**3. Maximum Iteration Limit**: To maintain computational efficiency and ensure bounded runtime, we impose a fixed upper bound on the number of iterations. In our implementation, we set this limit to 2 refinement rounds per input. We found this conservative limit sufficient for the majority of test cases, where either no vulnerabilities were detected after one pass or the second iteration resolved remaining issues.
