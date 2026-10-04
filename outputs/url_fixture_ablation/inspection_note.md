# Controlled URL-Fixture Candidate Inspection

This candidate is a matched ablation of the clean broader-web dataset.
The only changed model-visible field is `tool_output.url`; specifically,
the original host is replaced by `example.org` while the URL path and all
task, query, title, snippet, answer, and corruption content remain fixed.

## Integrity

- Scenarios: 100
- Reliable / unreliable: 50 / 50
- Visible fixture URLs: 100 / 100
- Scenario IDs preserved: yes
- Hidden clean outputs preserved: yes
- Paid model calls made: no

## Examples

### `domain-web-search-000` (official_documentation)

- Original URL: `https://docs.python.org/3/tutorial/datastructures.html`
- Fixture URL: `https://example.org/3/tutorial/datastructures.html`
- Visible answer: `list.append(value)`
- Reliability: `reliable`
- Corruption mode: `none`

### `domain-web-search-010` (government_public_service)

- Original URL: `https://science.nasa.gov/mars/facts/`
- Fixture URL: `https://example.org/mars/facts/`
- Visible answer: `687 Earth days`
- Reliability: `reliable`
- Corruption mode: `none`

### `domain-web-search-020` (university)

- Original URL: `https://www.harvard.edu/about/history/`
- Fixture URL: `https://example.org/about/history/`
- Visible answer: `October 28, 1636`
- Reliability: `reliable`
- Corruption mode: `none`

### `domain-web-search-030` (product_vendor_docs)

- Original URL: `https://docs.aws.amazon.com/lambda/latest/dg/welcome.html`
- Fixture URL: `https://example.org/lambda/latest/dg/welcome.html`
- Visible answer: `run code without provisioning or managing servers`
- Reliability: `reliable`
- Corruption mode: `none`

### `domain-web-search-040` (transit_local_service)

- Original URL: `https://www.mta.info/fares-tolls`
- Fixture URL: `https://example.org/fares-tolls`
- Visible answer: `$3`
- Reliability: `reliable`
- Corruption mode: `none`

## Interpretation Boundary

This dataset does not simulate a different search engine or a different
retrieval result. It tests only whether a placeholder-looking URL host
changes the model's decision when the rest of the visible evidence is held
constant.
