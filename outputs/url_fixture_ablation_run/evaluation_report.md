# Controlled URL-Fixture Ablation Results

All deltas are fixture URL minus real URL on the same scenarios.

| Model | Policy | Metric | Real | Fixture | Delta | 95% CI |
|---|---|---|---:|---:|---:|---:|
| gpt-5.4-mini | baseline | silent_acceptance | 12/50 (0.240) | 10/50 (0.200) | -0.040 | [-0.100, 0.000] |
| gpt-5.4-mini | baseline | false_defer_or_verify | 0/50 (0.000) | 0/50 (0.000) | +0.000 | [0.000, 0.000] |
| gpt-5.4-mini | baseline | trust_accuracy | 88/100 (0.880) | 90/100 (0.900) | +0.020 | [0.000, 0.050] |
| gpt-5.4-mini | m2_structured_skeptic | silent_acceptance | 4/50 (0.080) | 0/50 (0.000) | -0.080 | [-0.160, -0.020] |
| gpt-5.4-mini | m2_structured_skeptic | false_defer_or_verify | 1/50 (0.020) | 33/50 (0.660) | +0.640 | [0.500, 0.760] |
| gpt-5.4-mini | m2_structured_skeptic | trust_accuracy | 95/100 (0.950) | 67/100 (0.670) | -0.280 | [-0.350, -0.210] |
| gpt-5.4-mini | m6_checklist | silent_acceptance | 6/50 (0.120) | 0/50 (0.000) | -0.120 | [-0.220, -0.040] |
| gpt-5.4-mini | m6_checklist | false_defer_or_verify | 1/50 (0.020) | 30/50 (0.600) | +0.580 | [0.440, 0.720] |
| gpt-5.4-mini | m6_checklist | trust_accuracy | 93/100 (0.930) | 70/100 (0.700) | -0.230 | [-0.310, -0.140] |
| gemini-2.5-flash | baseline | silent_acceptance | 11/50 (0.220) | 11/50 (0.220) | +0.000 | [0.000, 0.000] |
| gemini-2.5-flash | baseline | false_defer_or_verify | 0/50 (0.000) | 0/50 (0.000) | +0.000 | [0.000, 0.000] |
| gemini-2.5-flash | baseline | trust_accuracy | 89/100 (0.890) | 89/100 (0.890) | +0.000 | [0.000, 0.000] |
| gemini-2.5-flash | m2_structured_skeptic | silent_acceptance | 8/50 (0.160) | 6/50 (0.120) | -0.040 | [-0.100, 0.000] |
| gemini-2.5-flash | m2_structured_skeptic | false_defer_or_verify | 0/50 (0.000) | 0/50 (0.000) | +0.000 | [0.000, 0.000] |
| gemini-2.5-flash | m2_structured_skeptic | trust_accuracy | 92/100 (0.920) | 94/100 (0.940) | +0.020 | [0.000, 0.050] |
| gemini-2.5-flash | m6_checklist | silent_acceptance | 10/50 (0.200) | 9/50 (0.180) | -0.020 | [-0.060, 0.000] |
| gemini-2.5-flash | m6_checklist | false_defer_or_verify | 0/50 (0.000) | 0/50 (0.000) | +0.000 | [0.000, 0.000] |
| gemini-2.5-flash | m6_checklist | trust_accuracy | 90/100 (0.900) | 91/100 (0.910) | +0.010 | [0.000, 0.030] |

Changed-action pairs available for review: 107.
Pre-registered clear-effect cells: 6.

Corruption-mode estimates are stored in `corruption_mode_deltas.json` and must be interpreted with their small denominators.
