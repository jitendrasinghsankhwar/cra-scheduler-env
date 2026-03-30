"""Grader: scores agent performance 0.0 to 1.0."""


def grade(sites_visited: int, total_sites: int, agent_cost: float, optimal_cost: float) -> float:
    """Score agent performance.

    Returns float between 0.0 and 1.0.
    Formula: 50% completion + 50% cost efficiency vs OR-Tools optimal.
    """
    if total_sites == 0:
        return 0.0

    completion = sites_visited / total_sites

    if agent_cost <= 0:
        efficiency = 0.0
    else:
        efficiency = min(1.0, optimal_cost / agent_cost)

    return round(completion * 0.5 + efficiency * 0.5, 4)
