# Reproducibility Artifact

This repository supports the paper **When Agents Trust the Wrong Tool
Output: Acceptance Across Corruption Types and a URL-Host Ablation**. It contains the two
paper-facing scenario sets, canonical model decisions, analysis code, prompt
builders, tests, and figure-generation code. Reproducing the reported metrics
does not require API credentials and does not make paid model calls.

**Authors:** Chirag Gupta and Rajeel Ahmad Ansari (Independent Researchers)

**Permanent repository:**
<https://github.com/20-Chirag/tool-output-trust-artifact>

## What Is Included

- `outputs/real_web_domain_expansion/`: 50 paired public-page captures,
  producing 50 reliable and 50 corrupted clean-web scenarios.
- `outputs/url_fixture_ablation/`: the matched URL-host treatment scenarios.
  Only the visible URL host is replaced with `example.org`.
- `outputs/*_run/runs.jsonl`: canonical, deduplicated model decisions. The
  clean-web response cache is also included because the qualitative diagnostic
  reads rationales from it; duplicate cache rows are resolved by event ID.
- `src/trustbench/`: prompt builders, policy definitions, and metric helpers.
- `scripts/paper/`: dataset audits and paper-facing evaluators.
- `paper/agent_tool_trust/generate_figures.py`: source for the two paper figures.
- `paper/agent_tool_trust/orx_figstyle.py`: vendored publication-figure style module;
  its MIT license and provenance are recorded in `THIRD_PARTY_LICENSES/`.
- `configs/`: exact model-policy matrices. Paid authorization is disabled.
- `tests/`: balance, pairing, leakage, and controlled-treatment checks.
- `outputs/phase_2/`, `outputs/phase_b/`, and
  `outputs/phase_3_minimal/`: the calculator decisions and robustness record
  supporting the explicitly secondary Appendix E diagnostic.
- `outputs/adjudication/claude_results_2026_07_22/` and
  `outputs/paper_qualitative_evidence/`: external-model labels and the
  dependence-aware evidence summary. These are not human annotations.
- `outputs/paper_claim_audit/`: the claim ledger and action/direct-use audit.

Other retry caches, API credentials, `.env` files, provider-account metadata,
and exploratory runs not used by the paper are intentionally excluded. The
single included clean-web cache is retained only because the secondary
qualitative diagnostic reads its recorded rationales.

The action-semantics JSON also records a broader 5,395-decision historical
audit across several project phases. The manuscript uses the directly
reconcilable paper scope: all seven conflicts occur in the 1,500-decision
clean-web run, so it reports 7/1,500.

## Environment

The code requires Python 3.10 or newer. From the archive root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .[dev]
$env:PYTHONPATH='src'
```

The supplied `uv.lock` records the project dependency resolution used in the
workspace. The paper evaluates hosted, versioned model APIs; exact replication
therefore also depends on continued provider access to the named model IDs.

On Linux or macOS, use the corresponding POSIX environment commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
export PYTHONPATH=src
```

## Validate The Scenario Design

These tests inspect only local files and make no network or model calls:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m pytest `
  tests\test_real_web_domain_expansion.py `
  tests\test_url_fixture_ablation.py -q
```

They verify the 50/50 reliability balance, 1:1 source pairing, corruption-mode
counts, absence of known leakage fields, public-domain constraints, and the
single-field URL-host intervention.

The equivalent Linux/macOS test command is:

```bash
PYTHONPATH=src python -m pytest \
  tests/test_real_web_domain_expansion.py \
  tests/test_url_fixture_ablation.py -q
```

## Reproduce The Paper-Facing Analyses

The following commands consume the included canonical decisions:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m scripts.paper.audit_clean_web_diagnostic
.\.venv\Scripts\python.exe -m scripts.paper.evaluate_url_fixture_ablation
.\.venv\Scripts\python.exe scripts\paper\validate_secondary_diagnostic.py --root .
.\.venv\Scripts\python.exe paper\agent_tool_trust\generate_figures.py
```

The first command regenerates corruption-mode metrics under
`outputs/clean_web_diagnostic/`. The second command regenerates the matched
URL-host effects and capture-level confidence intervals. The third verifies
every selected calculator label against its source case and the 0/26
stronger-model record. The final command writes the two vector figures and
preview files under `paper/agent_tool_trust/figures/`.

The equivalent Linux/macOS analysis commands are:

```bash
PYTHONPATH=src python -m scripts.paper.audit_clean_web_diagnostic
PYTHONPATH=src python -m scripts.paper.evaluate_url_fixture_ablation
PYTHONPATH=src python scripts/paper/validate_secondary_diagnostic.py --root .
PYTHONPATH=src python paper/agent_tool_trust/generate_figures.py
```

Simple proportions use two-sided 95% Wilson score intervals. Paired treatment
effects use 10,000 source-capture bootstrap resamples, preserving the reliable
and corrupted scenario pair and both URL conditions.

## Policy Names

The headline labels `M2` and `M6` refer to `m2_structured_skeptic` and
`m6_checklist`. The clean-web matrix also contains the diagnostic policies
`m0_naive_m2` and `m0_naive_m6`. Those are separate cells and are not used in
the paper's headline M2/M6 URL-host comparisons.

## Secondary Calculator Diagnostic

Appendix E is intentionally not one of the paper's two primary findings. Its
50 externally labeled decisions consist of 24 Phase 2 M6 decisions and 26 Phase
B M0-M6 decisions, drawn from 13 unique scenarios. The exact selected rows and
labels are in
`outputs/adjudication/claude_results_2026_07_22/f1_calculator_claude_review.csv`.
The original decision files and calculator scenarios are included in
`outputs/phase_2/` and `outputs/phase_b/`. The stronger-Gemini 0/26 robustness
record is in `outputs/phase_3_minimal/finding_robustness.json`.

The quote "Matches quick mental calculation" comes from the `m0_naive_m6`
checklist variant; the separate quote "Matches quick mental check exactly"
comes from `m6_checklist`. Both source decisions are included here.

These labels were produced by an external language model, not an independent
human annotator. The selected rows are repeated policy decisions rather than 50
independent scenarios. The manuscript therefore uses them only to illustrate
that a written rationale is not proof that a calculation occurred; it makes no
prevalence estimate or causal policy claim from these labels.

## Omission-Mode Boundary

Omission cases retain related visible context and are not used in either
primary contrast. In `domain-web-search-094`, the answer field is blanked but
the visible title contains "Ventra"; the included construct audit therefore
flags that accept as retained-context inference rather than a clean test of
whether the model noticed the missing answer field.

## Data Construction And Scope

Each source capture yields one reliable and one corrupted scenario. The 50
corrupted scenarios contain 19 author-constructed, type-compatible same-entity
false values, 15 type-compatible wrong-entity substitutions, and 16
partial-field omissions.
The visible prompt excludes the separate clean-answer field, reliability
label, corruption mode, source-lineage identifiers, and gold action.

The public-page records are curated search-result-like traces, not live
search-engine responses. Snippets are short author-written paraphrases grounded
in the recorded URLs. Corruptions are applied after capture. The benchmark is
intended for diagnostic research and is not a safety certification or a claim
about the reliability of any source website or deployed model.

## Integrity

`MANIFEST_SHA256.json` lists the SHA-256 digest and byte size of every archived
file except the manifest itself. See `ASSET_AND_LICENSE_NOTICE.md` for
provenance, third-party rights, and the current licensing status.

## Citation

Please cite the associated paper:

```text
Chirag Gupta and Rajeel Ahmad Ansari. "When Agents Trust the Wrong Tool
Output: Acceptance Across Corruption Types and a URL-Host Ablation." 2026.
```
