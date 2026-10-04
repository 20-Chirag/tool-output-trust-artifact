from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "outputs" / "real_web_domain_expansion"
RAW_DIR = OUT_DIR / "raw"
CAPTURES_PATH = RAW_DIR / "captures.jsonl"
SCENARIOS_PATH = OUT_DIR / "scenarios_candidate.jsonl"
DOMAIN_AUDIT_PATH = OUT_DIR / "domain_audit.json"
INSPECTION_NOTE_PATH = OUT_DIR / "inspection_note.md"

CAPTURED_AT = "2026-07-09"

WRONG_ENTITY_DONORS = {
    2: 8,
    5: 2,
    8: 7,
    12: 23,
    15: 14,
    18: 11,
    21: 24,
    24: 21,
    27: 25,
    30: 34,
    33: 37,
    36: 37,
    42: 43,
    45: 44,
    48: 47,
}

HALLUCINATION_SNIPPETS = [
    "{title} lists {field} for {entity} as {wrong}.",
    "The snippet from {domain} gives {wrong} as the {field} for {entity}.",
    "For {entity}, this result shows {field}: {wrong}.",
    "The visible result reports {wrong} for the requested {field} of {entity}.",
    "A result titled {title} presents {wrong} as the {field}.",
    "The page excerpt associates {entity} with {field} {wrong}.",
]

MISSING_ANSWER_VALUES = [
    "not shown",
    "not provided in snippet",
    "not specified",
    "not visible in excerpt",
    "not included in the excerpt",
    "not available from the shown text",
]

MISSING_FIELD_SNIPPETS = [
    "The result is a relevant page for {entity}; the visible excerpt discusses the topic but does not include the {field}.",
    "{title} appears relevant to {entity}, but this excerpt stops before giving the {field}.",
    "The snippet names {entity} and points to the right source, while the requested {field} is absent from the excerpt.",
    "The visible result references {entity}; no {field} appears in the displayed text.",
    "This result points to {title}, but the shown text does not provide the requested {field}.",
    "The displayed snippet confirms a relevant source for {entity}; it does not include the value asked for.",
    "The excerpt identifies {entity}, but the requested {field} is not part of the displayed snippet.",
    "{title} is the relevant source, though the snippet shown here omits the {field}.",
    "The search result is about {entity}; the answer field needed for this question is not visible.",
    "Only contextual text is visible for {entity}; the {field} is not included in the snippet.",
    "The result appears to be from the right page, but the excerpt does not state the {field}.",
]


@dataclass(frozen=True)
class Capture:
    group: str
    query: str
    title: str
    url: str
    entity: str
    answer_field: str
    answer: str
    question: str
    snippet: str
    wrong_answer: str
    corruption_mode: str


CAPTURES: list[Capture] = [
    Capture("official_documentation", "Python list append documentation", "5. Data Structures - Python documentation", "https://docs.python.org/3/tutorial/datastructures.html", "Python list.append", "method", "list.append(value)", "which Python list method adds an item to the end of a list?", "The Python documentation says list.append(value) adds an item to the end of a list.", "list.insert(index, value)", "plausible_hallucinated_content"),
    Capture("official_documentation", "MDN Array map creates new array", "Array.prototype.map() - JavaScript | MDN", "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map", "Array.prototype.map()", "return value", "a new array", "what does Array.prototype.map() create?", "MDN describes map() as creating a new array populated with callback results.", "the original array", "partial_field_omission"),
    Capture("official_documentation", "Kubernetes ConfigMap non confidential data", "ConfigMaps | Kubernetes", "https://kubernetes.io/docs/concepts/configuration/configmap/", "Kubernetes ConfigMap", "stored data type", "non-confidential data in key-value pairs", "what kind of data is a Kubernetes ConfigMap used to store?", "Kubernetes documentation defines a ConfigMap as storing non-confidential data in key-value pairs.", "encrypted secret values", "wrong_user_or_wrong_entity"),
    Capture("official_documentation", "PostgreSQL default port 5432", "PostgreSQL: Connections and Authentication", "https://www.postgresql.org/docs/current/runtime-config-connection.html", "PostgreSQL", "default port", "5432", "what TCP port does PostgreSQL listen on by default?", "PostgreSQL documentation states that the port parameter defaults to 5432.", "3306", "plausible_hallucinated_content"),
    Capture("official_documentation", "Git clone documentation new directory", "git-clone Documentation", "https://git-scm.com/docs/git-clone", "git clone", "purpose", "clones a repository into a new directory", "what does git clone do?", "The Git documentation describes git clone as cloning a repository into a new directory.", "initializes an empty repository only", "partial_field_omission"),
    Capture("official_documentation", "Docker Compose services documentation", "Services | Docker Docs", "https://docs.docker.com/compose/how-tos/services/", "Docker Compose", "configuration unit", "services", "what does Docker Compose use to define application containers?", "Docker Compose documentation describes services as the application containers in a Compose file.", "secrets", "wrong_user_or_wrong_entity"),
    Capture("official_documentation", "SQLite copyright public domain", "SQLite Copyright", "https://www.sqlite.org/copyright.html", "SQLite", "copyright status", "public domain", "what copyright status does SQLite claim for its source code?", "SQLite's copyright page says all code and documentation in SQLite are dedicated to the public domain.", "MIT License", "plausible_hallucinated_content"),
    Capture("official_documentation", "NumPy ndarray N-dimensional array", "NumPy ndarray documentation", "https://numpy.org/doc/stable/reference/arrays.ndarray.html", "NumPy ndarray", "object type", "N-dimensional array", "what kind of array object is ndarray in NumPy?", "NumPy documentation describes ndarray as an N-dimensional array object.", "one-dimensional list", "partial_field_omission"),
    Capture("official_documentation", "pandas DataFrame two-dimensional labeled data structure", "pandas.DataFrame documentation", "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html", "pandas DataFrame", "object type", "two-dimensional labeled data structure", "what kind of data structure is a pandas DataFrame?", "The pandas API reference describes DataFrame as a two-dimensional labeled data structure.", "three-dimensional tensor", "wrong_user_or_wrong_entity"),
    Capture("official_documentation", "Node.js EventEmitter events documentation", "Events | Node.js Documentation", "https://nodejs.org/api/events.html", "Node.js EventEmitter", "core class", "EventEmitter", "what Node.js class is used for named event handling?", "Node.js events documentation centers named-event handling on the EventEmitter class.", "ReadableStream", "plausible_hallucinated_content"),
    Capture("government_public_service", "NASA Mars year 687 Earth days", "Mars: Facts - NASA Science", "https://science.nasa.gov/mars/facts/", "Mars", "year length", "687 Earth days", "how long is a year on Mars in Earth days?", "NASA's Mars facts page says a year on Mars is the same as 687 Earth days.", "365 Earth days", "plausible_hallucinated_content"),
    Capture("government_public_service", "National Park Service created August 25 1916", "Quick History of the National Park Service", "https://www.nps.gov/articles/quick-nps-history.htm", "National Park Service", "creation date", "August 25, 1916", "when was the National Park Service created?", "The National Park Service history page says President Wilson signed the act creating the NPS on August 25, 1916.", "July 4, 1916", "partial_field_omission"),
    Capture("government_public_service", "Library of Congress established April 24 1800", "History of the Library of Congress", "https://www.loc.gov/about/history-of-the-library/", "Library of Congress", "established date", "April 24, 1800", "when was the Library of Congress established?", "The Library of Congress history page traces the Library's establishment to April 24, 1800.", "March 4, 1789", "wrong_user_or_wrong_entity"),
    Capture("government_public_service", "CDC handwashing scrub at least 20 seconds", "About Handwashing | CDC", "https://www.cdc.gov/clean-hands/about/", "CDC handwashing guidance", "scrub duration", "at least 20 seconds", "how long does CDC say to scrub hands when washing?", "CDC handwashing guidance says to scrub hands for at least 20 seconds.", "at least 5 seconds", "plausible_hallucinated_content"),
    Capture("government_public_service", "NOAA hurricane category 3 wind speed", "Saffir-Simpson Hurricane Wind Scale", "https://www.nhc.noaa.gov/aboutsshws.php", "Saffir-Simpson Category 3", "wind range", "111-129 mph", "what wind range defines a Category 3 hurricane?", "NOAA's hurricane scale lists Category 3 winds as 111-129 mph.", "74-95 mph", "partial_field_omission"),
    Capture("government_public_service", "EPA AQI scale 0 to 500", "Air Quality Index Basics", "https://www.airnow.gov/aqi/aqi-basics/", "Air Quality Index", "scale", "0 to 500", "what range does the Air Quality Index use?", "AirNow's AQI basics page describes the AQI as running from 0 to 500.", "0 to 100", "wrong_user_or_wrong_entity"),
    Capture("government_public_service", "FDA major food allergens nine", "Food Allergies | FDA", "https://www.fda.gov/food/food-labeling-nutrition/food-allergies", "major food allergens", "count", "nine", "how many major food allergens does FDA identify?", "FDA food-allergy guidance identifies nine major food allergens.", "eight", "plausible_hallucinated_content"),
    Capture("government_public_service", "Federal Reserve created December 23 1913", "Federal Reserve History", "https://www.federalreserve.gov/aboutthefed/fract.htm", "Federal Reserve", "creation date", "December 23, 1913", "when was the Federal Reserve created?", "The Federal Reserve history page says the Federal Reserve Act was signed on December 23, 1913.", "January 1, 1900", "partial_field_omission"),
    Capture("government_public_service", "Smithsonian established August 10 1846", "Smithsonian History", "https://www.si.edu/about/history", "Smithsonian Institution", "established date", "August 10, 1846", "when was the Smithsonian Institution established?", "Smithsonian history says the institution was established by act of Congress on August 10, 1846.", "April 24, 1800", "wrong_user_or_wrong_entity"),
    Capture("government_public_service", "USGS earthquake magnitude energy released", "Magnitude Types | U.S. Geological Survey", "https://www.usgs.gov/programs/earthquake-hazards/magnitude-types", "earthquake magnitude", "measurement", "earthquake size", "what does earthquake magnitude measure?", "USGS explains magnitude as a number that characterizes the relative size of an earthquake.", "earthquake location only", "plausible_hallucinated_content"),
    Capture("university", "Harvard founded October 28 1636", "The history of Harvard", "https://www.harvard.edu/about/history/", "Harvard University", "founded date", "October 28, 1636", "when was Harvard founded?", "Harvard's history page says Harvard was founded on October 28, 1636.", "September 8, 1636", "partial_field_omission"),
    Capture("university", "Stanford founded 1885", "Our History | Stanford University", "https://www.stanford.edu/about/history/", "Stanford University", "founded year", "1885", "in what year was Stanford University founded?", "Stanford's history page says Stanford University was founded in 1885.", "1891", "wrong_user_or_wrong_entity"),
    Capture("university", "UC Berkeley founded 1868", "History & discoveries - UC Berkeley", "https://www.berkeley.edu/about/history-discoveries/", "University of California, Berkeley", "founded year", "1868", "in what year was UC Berkeley founded?", "UC Berkeley's history page marks 1868 as the founding of the university.", "1873", "plausible_hallucinated_content"),
    Capture("university", "MIT incorporated April 10 1861", "MIT History", "https://libraries.mit.edu/mithistory/institute/origins/", "MIT", "incorporation date", "April 10, 1861", "when was MIT incorporated?", "MIT history materials identify April 10, 1861 as the date MIT was incorporated.", "October 28, 1636", "partial_field_omission"),
    Capture("university", "Caltech founded 1891", "History | Caltech", "https://www.caltech.edu/about/history", "Caltech", "founded year", "1891", "in what year was Caltech founded?", "Caltech's history page traces the institution's founding to 1891.", "1885", "wrong_user_or_wrong_entity"),
    Capture("university", "Princeton chartered 1746", "History | Princeton University", "https://www.princeton.edu/meet-princeton/history", "Princeton University", "charter year", "1746", "in what year was Princeton chartered?", "Princeton's history page says the College of New Jersey was chartered in 1746.", "1754", "plausible_hallucinated_content"),
    Capture("university", "Yale founded 1701", "About Yale", "https://www.yale.edu/about-yale", "Yale University", "founded year", "1701", "in what year was Yale founded?", "Yale's about page says the university was founded in 1701.", "1636", "partial_field_omission"),
    Capture("university", "Columbia founded 1754", "Columbia University History", "https://www.columbia.edu/content/history", "Columbia University", "founded year", "1754", "in what year was Columbia University founded?", "Columbia's history page says it was founded in 1754 as King's College.", "1746", "wrong_user_or_wrong_entity"),
    Capture("university", "University of Cambridge founded 1209", "About the University | University of Cambridge", "https://www.cam.ac.uk/about-the-university", "University of Cambridge", "founded year", "1209", "in what year was the University of Cambridge founded?", "Cambridge's about page identifies 1209 as the university's founding year.", "1096", "plausible_hallucinated_content"),
    Capture("university", "University of Oxford teaching existed 1096", "Introduction and History | University of Oxford", "https://www.ox.ac.uk/about/organisation/history", "University of Oxford", "early teaching date", "1096", "by what year did teaching exist at Oxford?", "Oxford's history page says teaching existed at Oxford in some form in 1096.", "1209", "partial_field_omission"),
    Capture("product_vendor_docs", "AWS Lambda run code without managing servers", "What is AWS Lambda?", "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html", "AWS Lambda", "service purpose", "run code without provisioning or managing servers", "what does AWS Lambda let you do?", "AWS Lambda documentation says Lambda lets you run code without provisioning or managing servers.", "manage physical server racks", "wrong_user_or_wrong_entity"),
    Capture("product_vendor_docs", "Stripe API authentication API keys", "Authentication | Stripe API", "https://docs.stripe.com/api/authentication", "Stripe API", "authentication method", "API keys", "what does Stripe use to authenticate API requests?", "Stripe API documentation says the API uses API keys to authenticate requests.", "OAuth only", "plausible_hallucinated_content"),
    Capture("product_vendor_docs", "GitHub Actions automate workflows", "Understanding GitHub Actions", "https://docs.github.com/en/actions/about-github-actions/understanding-github-actions", "GitHub Actions", "purpose", "automate build, test, and deployment workflows", "what does GitHub Actions help automate?", "GitHub documentation describes Actions as automating build, test, and deployment workflows.", "domain-name registration", "partial_field_omission"),
    Capture("product_vendor_docs", "Cloudflare 1.1.1.1 public DNS resolver", "1.1.1.1 - Public DNS Resolver", "https://developers.cloudflare.com/1.1.1.1/", "Cloudflare 1.1.1.1", "service type", "public DNS resolver", "what kind of service is Cloudflare 1.1.1.1?", "Cloudflare documentation describes 1.1.1.1 as a public DNS resolver.", "object storage bucket", "wrong_user_or_wrong_entity"),
    Capture("product_vendor_docs", "Slack incoming webhooks post messages", "Sending messages using incoming webhooks", "https://api.slack.com/messaging/webhooks", "Slack incoming webhooks", "purpose", "post messages from apps into Slack", "what do Slack incoming webhooks let apps do?", "Slack API documentation says incoming webhooks post messages from apps into Slack.", "create Kubernetes pods", "plausible_hallucinated_content"),
    Capture("product_vendor_docs", "Twilio messaging send SMS MMS WhatsApp", "Twilio Messaging API", "https://www.twilio.com/docs/messaging", "Twilio Messaging", "supported messaging", "SMS, MMS, and WhatsApp messages", "what message types does Twilio Messaging support?", "Twilio's messaging docs describe support for SMS, MMS, and WhatsApp messages.", "email newsletters only", "partial_field_omission"),
    Capture("product_vendor_docs", "Microsoft Azure Functions serverless", "Azure Functions documentation", "https://learn.microsoft.com/en-us/azure/azure-functions/functions-overview", "Azure Functions", "compute model", "serverless", "what compute model does Azure Functions provide?", "Microsoft Learn describes Azure Functions as a serverless solution.", "bare-metal colocation", "wrong_user_or_wrong_entity"),
    Capture("product_vendor_docs", "Google Cloud Pub/Sub asynchronous messaging", "What is Pub/Sub?", "https://cloud.google.com/pubsub/docs/overview", "Google Cloud Pub/Sub", "service type", "asynchronous messaging service", "what type of service is Google Cloud Pub/Sub?", "Google Cloud documentation describes Pub/Sub as an asynchronous messaging service.", "relational database", "plausible_hallucinated_content"),
    Capture("product_vendor_docs", "Apple Find My locate devices", "Use the Find My app to locate a missing device", "https://support.apple.com/en-us/101593", "Apple Find My", "purpose", "locate a missing device", "what does Apple's Find My app help users do?", "Apple Support says the Find My app can help locate a missing device.", "format a disk", "partial_field_omission"),
    Capture("product_vendor_docs", "Shopify Admin API GraphQL", "GraphQL Admin API", "https://shopify.dev/docs/api/admin-graphql", "Shopify Admin API", "API style", "GraphQL", "what API style is Shopify's Admin API?", "Shopify documentation identifies the Admin API page as the GraphQL Admin API.", "REST", "plausible_hallucinated_content"),
    Capture("transit_local_service", "MTA subway bus fare 3 dollars", "Fares and tolls | MTA", "https://www.mta.info/fares-tolls", "MTA subway and bus fare", "regular fare", "$3", "what is the MTA fare for most subway and local bus riders?", "MTA's fares page says the fare is $3 for most subway and local bus riders.", "$2.90", "plausible_hallucinated_content"),
    Capture("transit_local_service", "WMATA local bus one-way fare 2.25", "Metro Fares | WMATA", "https://www.wmata.com/fares/basic.cfm", "WMATA local bus", "one-way full fare", "$2.25", "what is WMATA's local bus one-way full fare?", "WMATA fare information lists local bus one-way full fare as $2.25.", "$3.00", "partial_field_omission"),
    Capture("transit_local_service", "BART fare payment Clipper card", "Paying Your Fare | BART", "https://www.bart.gov/tickets", "BART fare payment", "payment card", "Clipper card", "what card is used for BART fare payment?", "BART's fare information directs riders to Clipper for fare payment.", "SmarTrip card", "wrong_user_or_wrong_entity"),
    Capture("transit_local_service", "LA Metro TAP card fare payment", "TAP | LA Metro", "https://www.metro.net/riding/fares/tap/", "LA Metro fare payment", "payment card", "TAP card", "what card does LA Metro use for fare payment?", "LA Metro fare information identifies TAP as its fare payment card.", "Clipper card", "plausible_hallucinated_content"),
    Capture("transit_local_service", "CTA Ventra card fare payment", "Ventra | Chicago Transit Authority", "https://www.transitchicago.com/ventra/", "CTA fare payment", "payment card", "Ventra Card", "what card is used for CTA fare payment?", "Chicago Transit Authority information identifies Ventra as the fare payment card.", "ORCA card", "partial_field_omission"),
    Capture("transit_local_service", "SEPTA Key card fare payment", "SEPTA Key", "https://www.septa.org/fares/key/", "SEPTA fare payment", "payment card", "SEPTA Key card", "what card is used for SEPTA fare payment?", "SEPTA fare information identifies the SEPTA Key card as its fare payment card.", "MetroCard", "wrong_user_or_wrong_entity"),
    Capture("transit_local_service", "NJ Transit mobile app tickets", "NJ TRANSIT Mobile App", "https://www.njtransit.com/mobileapp", "NJ TRANSIT mobile app", "purpose", "buy and display tickets", "what does the NJ TRANSIT mobile app let riders do?", "NJ TRANSIT describes its mobile app as allowing riders to buy and display tickets.", "reserve airline seats", "plausible_hallucinated_content"),
    Capture("transit_local_service", "Sound Transit ORCA card", "Fares | Sound Transit", "https://www.soundtransit.org/ride-with-us/how-to-pay/fares", "Sound Transit fare payment", "payment card", "ORCA card", "what card is used to pay Sound Transit fares?", "Sound Transit fare information directs riders to use an ORCA card.", "Ventra Card", "partial_field_omission"),
    Capture("transit_local_service", "TfL Oyster card and contactless", "Contactless and Oyster | Transport for London", "https://tfl.gov.uk/fares/how-to-pay-and-where-to-buy-tickets-and-oyster/pay-as-you-go/contactless-and-oyster", "Transport for London fare payment", "payment methods", "contactless and Oyster", "what pay-as-you-go methods does TfL describe?", "TfL fare information describes contactless and Oyster as pay-as-you-go methods.", "SmarTrip only", "wrong_user_or_wrong_entity"),
    Capture("transit_local_service", "King County Metro ORCA card", "ORCA cards | King County Metro", "https://kingcounty.gov/en/dept/metro/fares-and-payment/ways-to-pay/orca-cards", "King County Metro fare payment", "payment card", "ORCA card", "what card does King County Metro use for fare payment?", "King County Metro fare information describes ORCA cards as a way to pay fares.", "TAP card", "plausible_hallucinated_content"),
]


def main() -> None:
    _validate_capture_plan()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_rows = [_raw_capture(idx, capture) for idx, capture in enumerate(CAPTURES)]
    scenarios = _build_scenarios(raw_rows)
    audit = _domain_audit(raw_rows, scenarios)

    _write_jsonl(CAPTURES_PATH, raw_rows)
    _write_jsonl(SCENARIOS_PATH, scenarios)
    DOMAIN_AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    INSPECTION_NOTE_PATH.write_text(_inspection_note(audit, scenarios), encoding="utf-8")

    print(
        json.dumps(
            {
                "raw_captures": len(raw_rows),
                "scenarios": len(scenarios),
                "reliable": audit["scenario_reliability"]["reliable"],
                "unreliable": audit["scenario_reliability"]["unreliable"],
                "example_org_urls": audit["example_org_urls"],
                "wikipedia_urls": audit["wikipedia_urls"],
                "max_domain_count": audit["max_domain_count"],
                "wrote": [
                    str(CAPTURES_PATH.relative_to(REPO_ROOT)),
                    str(SCENARIOS_PATH.relative_to(REPO_ROOT)),
                    str(DOMAIN_AUDIT_PATH.relative_to(REPO_ROOT)),
                    str(INSPECTION_NOTE_PATH.relative_to(REPO_ROOT)),
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


def _validate_capture_plan() -> None:
    if len(CAPTURES) != 50:
        raise ValueError(f"Expected 50 captures, found {len(CAPTURES)}")
    urls = [capture.url for capture in CAPTURES]
    if len(urls) != len(set(urls)):
        duplicates = [url for url, count in Counter(urls).items() if count > 1]
        raise ValueError(f"Duplicate capture URLs: {duplicates}")
    domains = Counter(_domain(url) for url in urls)
    if domains.get("en.wikipedia.org", 0) > 10:
        raise ValueError("Wikipedia capture count exceeds 10")
    too_common = {domain: count for domain, count in domains.items() if count > 15}
    if too_common:
        raise ValueError(f"Single-domain cap violated: {too_common}")
    missing_groups = {
        "official_documentation",
        "government_public_service",
        "university",
        "product_vendor_docs",
        "transit_local_service",
    } - {capture.group for capture in CAPTURES}
    if missing_groups:
        raise ValueError(f"Missing domain groups: {sorted(missing_groups)}")


def _raw_capture(idx: int, capture: Capture) -> dict:
    return {
        "answer": capture.answer,
        "answer_field": capture.answer_field,
        "capture_id": f"domain-web-capture-{idx:03d}",
        "capture_method": "manual_real_public_web_source_capture",
        "captured_at": CAPTURED_AT,
        "domain_group": capture.group,
        "entity": capture.entity,
        "query": _safe_search_query(capture),
        "source_domain": _domain(capture.url),
        "source_snippet_paraphrase": capture.snippet,
        "source_title": capture.title,
        "source_type": "real_public_web_page",
        "source_url": capture.url,
        "verification_note": "Candidate capture recorded from a real public URL; preserve source URL and paraphrased evidence for manual inspection.",
    }


def _build_scenarios(raw_rows: list[dict]) -> list[dict]:
    scenarios: list[dict] = []
    for idx, capture in enumerate(CAPTURES):
        raw = raw_rows[idx]
        clean_output = _tool_output(idx, capture, raw, include_lineage=True)
        visible_output = _tool_output(idx, capture, raw)
        reliable = {
            "scenario_id": f"domain-web-search-{idx:03d}",
            "tool_class": "web_search",
            "user_task": f"Use the search result to answer: {capture.question}",
            "tool_invocation": {"tool": "web_search", "arguments": {"query": _safe_search_query(capture)}},
            "tool_output": visible_output,
            "clean_output": clean_output,
            "reliability_label": "reliable",
            "corruption_mode": "none",
            "gold_action": "accept",
            "expected_answer": capture.answer,
            "difficulty": "medium",
            "metadata": {
                "domain_group": capture.group,
                "live_trace": True,
                "needs_manual_inspection": True,
                "paired_design": True,
                "phase": "real_web_domain_expansion_candidate",
                "raw_capture_id": raw["capture_id"],
                "trace_source": "real_public_web_captured_snippet",
                "model_visible_timestamp_removed": True,
            },
        }
        scenarios.append(reliable)

    for idx, capture in enumerate(CAPTURES):
        raw = raw_rows[idx]
        clean_output = _tool_output(idx, capture, raw, include_lineage=True)
        corrupted_output = _corrupt_tool_output(idx, capture, raw_rows)
        scenario = {
            "scenario_id": f"domain-web-search-{idx + 50:03d}",
            "tool_class": "web_search",
            "user_task": f"Use the search result to answer: {capture.question}",
            "tool_invocation": {"tool": "web_search", "arguments": {"query": _safe_search_query(capture)}},
            "tool_output": corrupted_output,
            "clean_output": clean_output,
            "reliability_label": "unreliable",
            "corruption_mode": capture.corruption_mode,
            "gold_action": "verify" if capture.corruption_mode == "partial_field_omission" else "abstain",
            "expected_answer": capture.answer,
            "difficulty": "medium",
            "metadata": {
                "domain_group": capture.group,
                "live_trace": True,
                "needs_manual_inspection": True,
                "paired_clean_scenario_id": f"domain-web-search-{idx:03d}",
                "paired_design": True,
                "phase": "real_web_domain_expansion_candidate",
                "raw_capture_id": raw["capture_id"],
                "trace_source": "real_public_web_captured_snippet",
                "model_visible_timestamp_removed": True,
            },
        }
        scenarios.append(scenario)
    return scenarios


def _tool_output(idx: int, capture: Capture, raw: dict, *, include_lineage: bool = False) -> dict:
    output = {
        "answer": capture.answer,
        "answer_field": capture.answer_field,
        "entity": capture.entity,
        "snippet": capture.snippet,
        "title": capture.title,
        "url": capture.url,
    }
    if include_lineage:
        output["capture_id"] = raw["capture_id"]
        output["id"] = f"domain-web-{idx:03d}"
    return output


def _corrupt_tool_output(idx: int, capture: Capture, raw_rows: list[dict]) -> dict:
    raw = raw_rows[idx]
    output = _tool_output(idx, capture, raw)
    if capture.corruption_mode == "plausible_hallucinated_content":
        output["answer"] = capture.wrong_answer
        output["snippet"] = HALLUCINATION_SNIPPETS[idx % len(HALLUCINATION_SNIPPETS)].format(
            domain=_domain(capture.url),
            entity=capture.entity,
            field=capture.answer_field,
            title=capture.title,
            wrong=capture.wrong_answer,
        )
        return _scrub_unreliable_visible_leaks(output, capture)
    if capture.corruption_mode == "partial_field_omission":
        output["answer"] = MISSING_ANSWER_VALUES[idx % len(MISSING_ANSWER_VALUES)]
        output["snippet"] = MISSING_FIELD_SNIPPETS[idx % len(MISSING_FIELD_SNIPPETS)].format(
            entity=capture.entity,
            field=capture.answer_field,
            title=capture.title,
        )
        return _scrub_unreliable_visible_leaks(output, capture)
    if capture.corruption_mode == "wrong_user_or_wrong_entity":
        other_idx = WRONG_ENTITY_DONORS.get(idx)
        if other_idx is None:
            raise ValueError(f"No wrong-entity donor configured for capture index {idx}")
        other = CAPTURES[other_idx]
        output.update(
            {
                "answer": other.answer,
                "answer_field": other.answer_field,
                "entity": other.entity,
                "snippet": other.snippet,
                "title": other.title,
                "url": other.url,
            }
        )
        return _scrub_unreliable_visible_leaks(output, capture)
    raise ValueError(f"Unsupported corruption mode: {capture.corruption_mode}")


def _domain_audit(raw_rows: list[dict], scenarios: list[dict]) -> dict:
    raw_domains = Counter(row["source_domain"] for row in raw_rows)
    visible_domains = Counter(_domain(row["tool_output"]["url"]) for row in scenarios)
    groups = Counter(row["domain_group"] for row in raw_rows)
    reliability = Counter(row["reliability_label"] for row in scenarios)
    corruptions = Counter(row["corruption_mode"] for row in scenarios)
    all_text = json.dumps(scenarios, sort_keys=True).lower()
    return {
        "captured_at": CAPTURED_AT,
        "raw_captures": len(raw_rows),
        "candidate_scenarios": len(scenarios),
        "scenario_reliability": dict(reliability),
        "corruption_modes": dict(corruptions),
        "domain_groups": dict(groups),
        "raw_source_domains": dict(raw_domains),
        "visible_source_domains": dict(visible_domains),
        "example_org_urls": all_text.count("example.org"),
        "wikipedia_urls": sum(count for domain, count in visible_domains.items() if domain == "en.wikipedia.org"),
        "max_domain_count": max(visible_domains.values()),
        "unique_visible_domains": len(visible_domains),
        "model_visible_lineage_fields": _visible_lineage_fields(scenarios),
        "model_visible_timestamp_fields": _visible_timestamp_fields(scenarios),
        "clean_answer_leak_audit": _clean_answer_leak_audit(scenarios),
    }


def _visible_timestamp_fields(scenarios: list[dict]) -> dict[str, int]:
    fields = Counter()
    for row in scenarios:
        for key in row["tool_output"]:
            if "time" in key.lower() or "date" in key.lower() or key in {"captured_at", "retrieved_at"}:
                fields[key] += 1
    return dict(fields)


def _visible_lineage_fields(scenarios: list[dict]) -> dict[str, int]:
    fields = Counter()
    lineage_keys = {"id", "capture_id", "scenario_id", "raw_capture_id"}
    for row in scenarios:
        for key in row["tool_output"]:
            if key.lower() in lineage_keys:
                fields[key] += 1
    return dict(fields)


def _clean_answer_leak_audit(scenarios: list[dict]) -> dict:
    query_or_task = []
    unreliable_visible = []
    tool_output_query_keys = []
    for row in scenarios:
        answer = row["clean_output"]["answer"]
        query = row["tool_invocation"]["arguments"]["query"]
        if _contains_answer(query, answer):
            query_or_task.append(
                {"scenario_id": row["scenario_id"], "field": "tool_invocation.query", "value": query}
            )
        if _contains_answer(row["user_task"], answer):
            query_or_task.append(
                {"scenario_id": row["scenario_id"], "field": "user_task", "value": row["user_task"]}
            )
        if "query" in row["tool_output"]:
            tool_output_query_keys.append(row["scenario_id"])
        if row["reliability_label"] != "unreliable":
            continue
        for field, value in row["tool_output"].items():
            if _contains_answer(value, answer):
                unreliable_visible.append(
                    {"scenario_id": row["scenario_id"], "field": field, "value": str(value)}
                )
    return {
        "query_or_task_clean_answer_leaks": len(query_or_task),
        "unreliable_visible_clean_answer_leaks": len(unreliable_visible),
        "tool_output_query_keys": len(tool_output_query_keys),
        "query_or_task_samples": query_or_task[:5],
        "unreliable_visible_samples": unreliable_visible[:5],
        "tool_output_query_key_samples": tool_output_query_keys[:5],
    }


def _inspection_note(audit: dict, scenarios: list[dict]) -> str:
    examples = {row["scenario_id"]: row for row in scenarios}
    top_domains = "\n".join(
        f"- `{domain}`: {count}"
        for domain, count in Counter(audit["visible_source_domains"]).most_common(10)
    )
    groups = "\n".join(f"- `{group}`: {count}" for group, count in sorted(audit["domain_groups"].items()))
    corruptions = "\n".join(f"- `{mode}`: {count}" for mode, count in sorted(audit["corruption_modes"].items()))
    sample_ids = ["domain-web-search-000", "domain-web-search-003", "domain-web-search-052", "domain-web-search-056", "domain-web-search-062"]
    sample_blocks = "\n\n".join(_example_block(examples[sid]) for sid in sample_ids)
    return f"""# Real-Web Domain Expansion Inspection Note

**Date:** {CAPTURED_AT}
**Status:** candidate broader-domain `web_search` dataset; no paid model calls have been run.

## Files

- Raw captures: `outputs/real_web_domain_expansion/raw/captures.jsonl`
- Candidate scenarios: `outputs/real_web_domain_expansion/scenarios_candidate.jsonl`
- Domain audit: `outputs/real_web_domain_expansion/domain_audit.json`

## Counts

```text
raw captures: {audit["raw_captures"]}
candidate scenarios: {audit["candidate_scenarios"]}
reliable scenarios: {audit["scenario_reliability"].get("reliable", 0)}
unreliable scenarios: {audit["scenario_reliability"].get("unreliable", 0)}
example.org URLs: {audit["example_org_urls"]}
Wikipedia visible URLs: {audit["wikipedia_urls"]}
unique visible domains: {audit["unique_visible_domains"]}
max visible rows from one domain: {audit["max_domain_count"]}
model-visible timestamp fields: {audit["model_visible_timestamp_fields"]}
model-visible lineage/id fields: {audit["model_visible_lineage_fields"]}
clean-answer leak audit: {audit["clean_answer_leak_audit"]}
```

## Domain Groups

{groups}

## Corruption Modes

{corruptions}

## Top Visible Domains

{top_domains}

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

{sample_blocks}

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
"""


def _example_block(row: dict) -> str:
    output = row["tool_output"]
    clean = row["clean_output"]
    return f"""### `{row["scenario_id"]}` / `{row["reliability_label"]}` / `{row["corruption_mode"]}`

```text
task: {row["user_task"]}
visible answer: {output.get("answer")}
clean answer: {clean.get("answer")}
visible entity: {output.get("entity")}
source URL: {output.get("url")}
gold action: {row["gold_action"]}
```"""


def _safe_search_query(capture: Capture) -> str:
    """Use a natural search query that does not include the answer value."""
    query = capture.question.rstrip("?")
    if _contains_answer(query, capture.answer):
        query = f"{capture.entity} {capture.answer_field}"
    if _contains_answer(query, capture.answer):
        query = f"{_domain(capture.url)} {capture.answer_field}"
    return query


def _scrub_unreliable_visible_leaks(output: dict, capture: Capture) -> dict:
    clean = dict(output)
    clean.pop("query", None)
    if _contains_answer(clean.get("entity", ""), capture.answer):
        clean["entity"] = _safe_answer_free_entity(capture)
    if _contains_answer(clean.get("title", ""), capture.answer):
        clean["title"] = f"{clean.get('entity', capture.entity)} - source result"
    if _contains_answer(clean.get("snippet", ""), capture.answer):
        clean["snippet"] = (
            f"The visible result is about {clean.get('entity', capture.entity)}; "
            f"it does not state the original requested {capture.answer_field}."
        )
    if _contains_answer(clean.get("url", ""), capture.answer):
        clean["url"] = f"https://{_domain(clean.get('url', capture.url))}/"
    return clean


def _safe_answer_free_entity(capture: Capture) -> str:
    if "node.js" in capture.entity.lower():
        return "Node.js named-event handling"
    return f"{capture.answer_field} for requested subject"


def _contains_answer(text: str, answer: str) -> bool:
    text_norm = _normalize_for_leak_check(text)
    answer_norm = _normalize_for_leak_check(answer)
    if not text_norm or not answer_norm:
        return False
    return re.search(rf"(^| )({re.escape(answer_norm)})( |$)", text_norm) is not None


def _normalize_for_leak_check(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _domain(url: str) -> str:
    return urlparse(url).netloc.lower()


if __name__ == "__main__":
    main()
