"""Grader: scores agent performance on a 0–1 scale.

Formula: 50% completion (sites visited / total) + 50% efficiency (optimal / agent cost).
Scores are clamped to the open interval (0, 1) as required by the hackathon validator.
"""


def grade(sites_visited: int, total_sites: int, agent_cost: float, optimal_cost: float) -> float:
    """Score agent performance.

    Args:
        sites_visited: Number of sites successfully visited within their time window.
        total_sites: Total number of sites in the task.
        agent_cost: Total travel cost (miles) incurred by the agent.
        optimal_cost: OR-Tools optimal cost for the same task.

    Returns:
        Score in (0.001, 0.999) — strictly between 0 and 1.
    """
    if total_sites == 0:
        return 0.001

    completion = sites_visited / total_sites

    if agent_cost <= 0:
        efficiency = 0.0
    else:
        efficiency = min(1.0, optimal_cost / agent_cost)

    raw = round(completion * 0.5 + efficiency * 0.5, 4)
    return min(max(raw, 0.001), 0.999)
