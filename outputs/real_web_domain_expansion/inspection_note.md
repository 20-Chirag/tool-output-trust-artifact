# Real-Web Domain Expansion Inspection Note

**Date:** 2026-07-09
**Status:** candidate broader-domain `web_search` dataset; no paid model calls have been run.

## Files

- Raw captures: `outputs/real_web_domain_expansion/raw/captures.jsonl`
- Candidate scenarios: `outputs/real_web_domain_expansion/scenarios_candidate.jsonl`
- Domain audit: `outputs/real_web_domain_expansion/domain_audit.json`

## Counts

```text
raw captures: 50
candidate scenarios: 100
reliable scenarios: 50
unreliable scenarios: 50
example.org URLs: 0
Wikipedia visible URLs: 0
unique visible domains: 50
max visible rows from one domain: 4
model-visible timestamp fields: {}
model-visible lineage/id fields: {}
clean-answer leak audit: {'query_or_task_clean_answer_leaks': 0, 'unreliable_visible_clean_answer_leaks': 0, 'tool_output_query_keys': 0, 'query_or_task_samples': [], 'unreliable_visible_samples': [], 'tool_output_query_key_samples': []}
```

## Domain Groups

- `government_public_service`: 10
- `official_documentation`: 10
- `product_vendor_docs`: 10
- `transit_local_service`: 10
- `university`: 10

## Corruption Modes

- `none`: 50
- `partial_field_omission`: 16
- `plausible_hallucinated_content`: 19
- `wrong_user_or_wrong_entity`: 15

## Top Visible Domains

- `cloud.google.com`: 4
- `numpy.org`: 3
- `www.nps.gov`: 3
- `www.nhc.noaa.gov`: 3
- `libraries.mit.edu`: 3
- `www.princeton.edu`: 3
- `api.slack.com`: 3
- `www.metro.net`: 3
- `www.transitchicago.com`: 3
- `www.soundtransit.org`: 3

## Design Notes

- This is a paired design: each of 50 raw captures creates one reliable row and
  one unreliable row.
- Visible `tool_output` rows include the answer, answer field, title, URL,
  entity, and snippet paraphrase. The search query appears only in
  `tool_invocation.arguments.query`.
- Visible `tool_output` rows intentionally exclude `id`, `capture_id`, and
  other lineage fields, after external review caught that an earlier candidate
  encoded corruption type in visible IDs.
- Capture timestamps are preserved in raw capture metadata, but are intentionally
  removed from model-visible `tool_output` to avoid the future/stale timestamp
  confound observed in earlier web-search diagnostics.
- The candidate keeps old Phase 2 and unified Wikipedia-only outputs untouched.

## Example Cases

### `domain-web-search-000` / `reliable` / `none`

```text
task: Use the search result to answer: which Python list method adds an item to the end of a list?
visible answer: list.append(value)
clean answer: list.append(value)
visible entity: Python list.append
source URL: https://docs.python.org/3/tutorial/datastructures.html
gold action: accept
```

### `domain-web-search-003` / `reliable` / `none`

```text
task: Use the search result to answer: what TCP port does PostgreSQL listen on by default?
visible answer: 5432
clean answer: 5432
visible entity: PostgreSQL
source URL: https://www.postgresql.org/docs/current/runtime-config-connection.html
gold action: accept
```

### `domain-web-search-052` / `unreliable` / `wrong_user_or_wrong_entity`

```text
task: Use the search result to answer: what kind of data is a Kubernetes ConfigMap used to store?
visible answer: two-dimensional labeled data structure
clean answer: non-confidential data in key-value pairs
visible entity: pandas DataFrame
source URL: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
gold action: abstain
```

### `domain-web-search-056` / `unreliable` / `plausible_hallucinated_content`

```text
task: Use the search result to answer: what copyright status does SQLite claim for its source code?
visible answer: MIT License
clean answer: public domain
visible entity: SQLite
source URL: https://www.sqlite.org/copyright.html
gold action: abstain
```

### `domain-web-search-062` / `unreliable` / `wrong_user_or_wrong_entity`

```text
task: Use the search result to answer: when was the Library of Congress established?
visible answer: April 10, 1861
clean answer: April 24, 1800
visible entity: MIT
source URL: https://libraries.mit.edu/mithistory/institute/origins/
gold action: abstain
```

## Inspection Verdict

This candidate satisfies the planned no-paid dataset criteria:

- 100 web-search scenarios;
- 50 reliable and 50 unreliable;
- zero `example.org`;
- zero Wikipedia rows;
- no single visible domain exceeds 15 rows;
- mixed official documentation, government/public-service, university,
  product/vendor documentation, and transit/local-service domains.

Before any paid model calls, run the focused tests and the dry-run cost estimate
from `configs/real_web_domain_expansion.yaml`.
