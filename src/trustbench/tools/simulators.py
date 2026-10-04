from __future__ import annotations

from trustbench.data.schemas import Scenario


def verify_with_second_tool(scenario: Scenario) -> dict:
    """Return an independent clean response for the same task.

    The benchmark treats this as the upper-level capability to cross-check a
    suspicious result. Real deployments can replace this with another search
    query, code execution engine, API endpoint, or retrieval index.
    """

    verified = dict(scenario.clean_output)
    verified["verification_source"] = "independent_simulated_tool"
    return verified
