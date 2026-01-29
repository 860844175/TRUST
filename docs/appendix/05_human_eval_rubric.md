# Human Evaluation Rubric for TRUST-Bench

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
