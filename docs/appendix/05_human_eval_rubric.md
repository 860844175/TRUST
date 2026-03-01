# Human Evaluation Rubric for TRUST-Bench

<a id="tab:rubric_trustbench"></a>
| **Evaluation Aspect** | **Assessment Criteria** | **Outcome** |
| :--- | :--- | :--- |
| Vulnerability Mask Accuracy | The automatically generated vulnerability mask correctly covers the vulnerable code regions modified or fixed by the commit, and the marked region is sufficiently precise without excessively extending beyond the true vulnerable code span. | Pass / Fail |
| Explanation--Code Alignment | The natural language explanation accurately describes the root cause of the vulnerability and is consistent with the semantic changes introduced by the corresponding commit. | Pass / Fail |
| Explanation Completeness | The explanation captures the key security-relevant aspects of the vulnerability, such as triggering conditions, consequences, or misuse patterns, without omitting critical information. | Pass / Fail |
| Semantic Consistency | The vulnerability mask, explanation, and code diff are mutually consistent and describe the same vulnerability instance without contradiction. | Pass / Fail |
| Overall Sample Validity | After considering all criteria above, the sample is deemed suitable for use as reliable supervision in downstream training and evaluation. | Retain / Revise / Remove |

*Caption: Human Evaluation Rubric for TRUST-Bench Quality Control.*

To ensure the quality and reliability of TRUST-Bench, we conduct a structured human evaluation based on the rubric in the table referenced above.
The rubric evaluates five aspects critical for vulnerability-aware supervision: vulnerability mask accuracy, explanation–code alignment, explanation completeness, semantic consistency, and overall sample validity.
Each criterion is assessed using binary decisions (Pass/Fail) to minimize subjectivity, while overall validity determines whether a sample is retained, revised, or removed.

The evaluation is independently performed by three security experts.
Expert E1 holds a Ph.D. in computer security with over three years of experience in vulnerability analysis and secure software development.
Expert E2 is a senior security engineer with extensive industry experience in auditing real-world vulnerabilities and analyzing security patches.
Expert E3 is a doctoral researcher specializing in program analysis and vulnerability detection, with prior experience in constructing security benchmarks.
All experts annotate independently without access to each other’s results.

Inter-annotator agreement is measured using Fleiss’ Kappa. The overall agreement reaches $\kappa = 0.68$, indicating substantial agreement, with per-criterion Kappa values ranging from $0.56$ to $0.77$. Higher agreement is observed for vulnerability mask accuracy and semantic consistency. Disagreements are resolved through majority voting, and samples failing multiple criteria are revised or removed, ensuring reliable benchmark construction.
