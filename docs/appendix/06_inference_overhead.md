# Inference Overhead Analysis

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
