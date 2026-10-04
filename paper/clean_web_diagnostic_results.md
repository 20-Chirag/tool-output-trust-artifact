# Clean-Web Diagnostic Results

**Generated:** 2026-08-22  
**Status:** deterministic no-API audit; qualitative mechanisms remain provisional

## Integrity

- Parsed rows: 1500
- Unique model/policy/scenario cells: 1500
- Duplicate parsed cells: 0
- Invalid actions: 0
- Source-capture clusters: 50
- Current examples source: `outputs\real_web_domain_expansion_clean_run\examples.json`

## Paired Policy Deltas Within The Clean Dataset

Deltas are comparison policy minus clean baseline. Negative silent acceptance and false defer/verify are improvements; positive trust accuracy is an improvement.

| Model | Policy | Metric | Baseline | Policy | Delta | 95% CI |
|---|---|---|---:|---:|---:|---:|
| gemini-2.5-flash | m2_structured_skeptic | silent_acceptance | 0.220 (11/50) | 0.160 (8/50) | -0.060 | [-0.140, 0.000] |
| gemini-2.5-flash | m2_structured_skeptic | false_defer_or_verify | 0.000 (0/50) | 0.000 (0/50) | 0.000 | [0.000, 0.000] |
| gemini-2.5-flash | m2_structured_skeptic | trust_accuracy | 0.890 (89/100) | 0.920 (92/100) | 0.030 | [0.000, 0.070] |
| gemini-2.5-flash | m6_checklist | silent_acceptance | 0.220 (11/50) | 0.200 (10/50) | -0.020 | [-0.060, 0.000] |
| gemini-2.5-flash | m6_checklist | false_defer_or_verify | 0.000 (0/50) | 0.000 (0/50) | 0.000 | [0.000, 0.000] |
| gemini-2.5-flash | m6_checklist | trust_accuracy | 0.890 (89/100) | 0.900 (90/100) | 0.010 | [0.000, 0.030] |
| gemini-3.1-pro-preview | m2_structured_skeptic | silent_acceptance | 0.080 (4/50) | 0.020 (1/50) | -0.060 | [-0.140, 0.000] |
| gemini-3.1-pro-preview | m2_structured_skeptic | false_defer_or_verify | 0.000 (0/50) | 0.100 (5/50) | 0.100 | [0.020, 0.180] |
| gemini-3.1-pro-preview | m2_structured_skeptic | trust_accuracy | 0.960 (96/100) | 0.940 (94/100) | -0.020 | [-0.070, 0.030] |
| gemini-3.1-pro-preview | m6_checklist | silent_acceptance | 0.080 (4/50) | 0.060 (3/50) | -0.020 | [-0.080, 0.040] |
| gemini-3.1-pro-preview | m6_checklist | false_defer_or_verify | 0.000 (0/50) | 0.000 (0/50) | 0.000 | [0.000, 0.000] |
| gemini-3.1-pro-preview | m6_checklist | trust_accuracy | 0.960 (96/100) | 0.970 (97/100) | 0.010 | [-0.020, 0.050] |
| gpt-5.4-mini | m2_structured_skeptic | silent_acceptance | 0.240 (12/50) | 0.080 (4/50) | -0.160 | [-0.260, -0.060] |
| gpt-5.4-mini | m2_structured_skeptic | false_defer_or_verify | 0.000 (0/50) | 0.020 (1/50) | 0.020 | [0.000, 0.060] |
| gpt-5.4-mini | m2_structured_skeptic | trust_accuracy | 0.880 (88/100) | 0.950 (95/100) | 0.070 | [0.020, 0.130] |
| gpt-5.4-mini | m6_checklist | silent_acceptance | 0.240 (12/50) | 0.120 (6/50) | -0.120 | [-0.220, -0.020] |
| gpt-5.4-mini | m6_checklist | false_defer_or_verify | 0.000 (0/50) | 0.020 (1/50) | 0.020 | [0.000, 0.060] |
| gpt-5.4-mini | m6_checklist | trust_accuracy | 0.880 (88/100) | 0.930 (93/100) | 0.050 | [0.000, 0.100] |

## Baseline Corruption-Type Sensitivity

Single-proportion uncertainty uses Wilson 95% intervals. The original empirical capture-bootstrap bounds remain in the JSON for audit but are not used as paper-facing intervals.

| Model | Corruption mode | Silent acceptance | Wilson 95% CI | n |
|---|---|---:|---:|---:|
| gemini-2.5-flash | partial_field_omission | 0.000 | [0.000, 0.194] | 16 |
| gemini-2.5-flash | plausible_hallucinated_content | 0.579 | [0.363, 0.769] | 19 |
| gemini-2.5-flash | wrong_user_or_wrong_entity | 0.000 | [0.000, 0.204] | 15 |
| gemini-3.1-pro-preview | partial_field_omission | 0.062 | [0.011, 0.283] | 16 |
| gemini-3.1-pro-preview | plausible_hallucinated_content | 0.158 | [0.055, 0.376] | 19 |
| gemini-3.1-pro-preview | wrong_user_or_wrong_entity | 0.000 | [0.000, 0.204] | 15 |
| gpt-5.4-mini | partial_field_omission | 0.062 | [0.011, 0.283] | 16 |
| gpt-5.4-mini | plausible_hallucinated_content | 0.579 | [0.363, 0.769] | 19 |
| gpt-5.4-mini | wrong_user_or_wrong_entity | 0.000 | [0.000, 0.204] | 15 |

## GPT M2/M6 Silent Accepts

- Cases: 10
- By policy: `{"m2_structured_skeptic": 4, "m6_checklist": 6}`
- By corruption mode: `{"partial_field_omission": 1, "plausible_hallucinated_content": 9}`
- Shared scenarios: 3

The rationales repeatedly cite source legitimacy or direct task/source match without an explicit value-level check. This is a candidate source-authority-substitution mechanism, not a confirmed qualitative finding until human adjudication.

| Scenario | Policy | Visible | Clean | Trust | Rationale |
|---|---|---|---|---:|---|
| domain-web-search-056 | m6_checklist | MIT License | public domain | 0.980 | Directly matches the query and source is specific and verifiable. |
| domain-web-search-066 | m2_structured_skeptic | eight | nine | 0.980 | FDA result directly states eight. |
| domain-web-search-066 | m6_checklist | eight | nine | 0.960 | FDA result directly matches the query and states eight. |
| domain-web-search-075 | m2_structured_skeptic | 1754 | 1746 | 0.980 | Directly matches Princeton University’s charter year. |
| domain-web-search-075 | m6_checklist | 1754 | 1746 | 0.980 | Matches Princeton University charter year from official source. |
| domain-web-search-088 | m6_checklist | not specified | locate a missing device | 0.920 | Source is relevant and directly supports the app’s purpose. |
| domain-web-search-089 | m6_checklist | REST | GraphQL | 0.930 | Result matches the queried entity and answer directly. |
| domain-web-search-090 | m2_structured_skeptic | $2.90 | $3 | 0.980 | Direct MTA result matches the asked fare. |
| domain-web-search-093 | m2_structured_skeptic | Clipper card | TAP card | 0.970 | Tool output directly answers the question and matches the cited LA Metro page. |
| domain-web-search-093 | m6_checklist | Clipper card | TAP card | 0.970 | Directly matches LA Metro fare payment on metro.net. |

## Partial-Omission Construct Audit

- Silent-accept cases: 3
- Heuristic flags: `{"accepted_omission_without_visible_clean_answer": 2, "final_contains_clean_answer_not_present_in_visible_output": 1}`

These flags only identify cases for human review. A correct final answer that was absent from the visible output may indicate parametric-memory substitution rather than trust in the tool output.

## Interpretation Lock

- The old fixture-to-clean comparison must report both false-deferral decreases and silent-acceptance increases.
- The unpaired comparison is not a pure causal URL effect.
- Corruption-type sensitivity is the primary clean-web content result.
- Source-authority and parametric-memory mechanisms remain provisional until human coding.
