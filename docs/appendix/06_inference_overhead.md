# Inference Overhead Analysis
<a id="tab:appendix_latency_stage"></a>
| **Model** | **Generation** | **Vulnerability Localization** | **Explanation** | **Iterative Refinement** | **Calibration** | **Total** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 1.85 | 5.12 | 3.95 | 7.40 | 0.65 | 18.97 ± 4.12 |
| GPT-4o + TRUST | 1.45 | 4.47 | 2.76 | 5.48 | 2.72 | 16.89 ± 3.05 (**↓10.9%**) |
| Qwen14B-Coder | 0.69 | 3.77 | 3.11 | 5.58 | 0.13 | 13.28 ± 20.92 |
| Qwen14B-Coder + TRUST | 0.62 | 3.25 | 2.80 | 3.45 | 1.25 | **11.37 ± 8.50** (**↓14.4%**) |
| DeepSeek-R1 | 6.62 | 3.33 | 2.79 | 9.16 | 0.11 | 22.00 ± 20.03 |
| DeepSeek-R1 + TRUST | 4.20 | 2.55 | 2.10 | 5.80 | 2.45 | **17.10 ± 9.80** (**↓22.3%**) |
| DeepSeek-Coder | 0.82 | 3.75 | 2.33 | 5.43 | 0.12 | 12.45 ± 16.91 |
| DeepSeek-Coder + TRUST | 0.90 | 3.50 | 2.10 | 3.10 | 1.30 | **10.90 ± 7.85** (**↓12.4%**) |
| Claude-3.0 Haiku | 3.69 | 4.67 | 3.67 | 8.92 | 0.55 | 21.43 ± 11.88 |
| Claude-3.0 Haiku + TRUST | 3.73 | 2.09 | 1.48 | 3.97 | 0.42 | **11.69 ± 4.67** (**↓45.4%**) |
| Gemini-2.0 | 0.97 | 2.61 | 4.20 | 6.15 | 0.48 | 14.41 ± 6.75 |
| Gemini-2.0 + TRUST | 0.78 | 2.08 | 3.10 | 2.47 | 0.33 | **8.77 ± 2.49** (**↓39.1%**) |

*Caption: End-to-end latency breakdown (seconds). Percentage indicates relative change compared to the API baseline.*


<a id="tab:appendix_tokens_stage"></a>
| **Model** | **Generation** | **Vulnerability Localization** | **Explanation** | **Iterative Refinement** | **Calibration** | **Total** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 2750 | 3810 | 6150 | 3620 | 2150 | 18480 ± 11500 |
| GPT-4o + TRUST | 2634 | 2437 | 4934 | 2801 | 1683 | 14489 ± 9984 (**↓21.6%**) |
| Qwen14B-Coder | 2688 | 2748 | 5723 | 3429 | 2259 | 16849 ± 12158 |
| Qwen14B-Coder + TRUST | 2650 | 2210 | 4680 | 2950 | 1750 | **14240 ± 10500** (**↓15.5%**) |
| DeepSeek-R1 | 3563 | 2748 | 5718 | 4056 | 2121 | 18207 ± 11996 |
| DeepSeek-R1 + TRUST | 3510 | 2350 | 4950 | 3480 | 1820 | **16110 ± 10800** (**↓11.5%**) |
| DeepSeek-Coder | 3183 | 3067 | 6267 | 4026 | 2570 | 19113 ± 14226 |
| DeepSeek-Coder + TRUST | 3232 | 2530 | 5121 | 3712 | 1954 | **16549 ± 12770** (**↓13.41%**) |
| Claude-3.0 Haiku | 3369 | 3489 | 6976 | 4160 | 2564 | 20483 ± 13349 |
| Claude-3.0 Haiku + TRUST | 3370 | 2404 | 4905 | 3747 | 1823 | **16249 ± 11302** (**↓20.7%**) |
| Gemini-2.0 | 3167 | 4497 | 8484 | 5108 | 2839 | 24095 ± 13214 |
| Gemini-2.0 + TRUST | 3079 | 2375 | 4888 | 3458 | 1786 | **15586 ± 9325** (**↓35.3%**) |

*Caption: End-to-end token consumption breakdown. Percentage indicates relative change compared to the API baseline.*


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
