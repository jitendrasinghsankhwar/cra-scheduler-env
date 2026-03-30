"""Fixed task definitions for the 3 difficulty levels.

Design principles:
- Easy: all sites reachable, wide windows, nearby cities, 1 CRA
- Medium: all sites reachable but requires smart CRA assignment, mixed windows
- Hard: not all sites reachable in time, forces tradeoff decisions
"""

import random

try:
    from .distances import ALL_CITIES
except ImportError:
    from distances import ALL_CITIES


def _generate_sites(cities: list, num_sites: int, day_range: tuple, window_size_range: tuple, seed: int, add_priority: bool = False) -> list:
    """Generate fixed site list with time windows. Deterministic via seed."""
    rng = random.Random(seed)
    selected = rng.sample(cities, min(num_sites, len(cities)))
    sites = []
    for name in selected:
        start = rng.randint(day_range[0], day_range[1])
        window_size = rng.randint(window_size_range[0], window_size_range[1])
        site = {
            "name": name,
            "window_start": start,
            "window_end": start + window_size,
        }
        if add_priority:
            site["priority"] = rng.choice(["critical", "high", "normal"])
        sites.append(site)
    return sites


_EASY_CRA_HOMES = ["Trenton, NJ"]

_MEDIUM_CRA_HOMES = ["Trenton, NJ", "Chicago, IL", "Atlanta, GA"]

_HARD_CRA_HOMES = [
    "Trenton, NJ", "Baltimore, MD", "Chicago, IL", "Dallas, TX",
    "Los Angeles, CA", "Seattle, WA", "Atlanta, GA", "Denver, CO",
    "Boston, MA", "Miami, FL",
]


def _available_sites(exclude: list) -> list:
    return [c for c in ALL_CITIES if c not in exclude]


# Northeast cities only for easy task (short distances)
_NORTHEAST = [
    "Philadelphia, PA", "NYC, NY", "Newark, NJ", "Wilmington, DE",
    "Baltimore, MD", "Washington, DC", "Hartford, CT", "New Haven, CT",
    "Providence, RI", "Boston, MA", "Albany, NY", "Atlantic City, NJ",
]

TASKS = {
    # EASY: 1 CRA, 4 nearby northeast sites, very wide windows
    # A greedy agent should score ~0.95+
    # All sites easily reachable within windows
    "easy": {
        "task_id": "easy",
        "description": (
            "1 CRA based in Trenton, NJ must visit 4 nearby sites in the Northeast. "
            "All time windows are wide (15-25 days). Any reasonable route works. "
            "Optimize for minimum travel distance."
        ),
        "cras": [{"id": 0, "home_city": "Trenton, NJ"}],
        "sites": [
            {"name": "Philadelphia, PA", "window_start": 1, "window_end": 25},
            {"name": "NYC, NY", "window_start": 1, "window_end": 20},
            {"name": "Newark, NJ", "window_start": 1, "window_end": 25},
            {"name": "Wilmington, DE", "window_start": 1, "window_end": 20},
        ],
        "max_days": 30,
    },

    # MEDIUM: 3 CRAs, 15 sites across East/Midwest, mixed windows
    # Requires smart assignment of sites to CRAs
    # Some windows are tight (4-6 days), some loose (8-14 days)
    # A greedy agent should score ~0.75-0.85
    "medium": {
        "task_id": "medium",
        "description": (
            "3 CRAs based in Trenton NJ, Chicago IL, and Atlanta GA must visit "
            "15 sites across the Eastern US and Midwest. Time windows vary from "
            "tight (4 days) to loose (14 days). Smart assignment of sites to the "
            "nearest CRA is critical. Some sites have overlapping windows forcing "
            "prioritization decisions."
        ),
        "cras": [
            {"id": 0, "home_city": "Trenton, NJ"},
            {"id": 1, "home_city": "Chicago, IL"},
            {"id": 2, "home_city": "Atlanta, GA"},
        ],
        "sites": [
            # Northeast cluster (CRA 0 territory)
            {"name": "Boston, MA", "window_start": 1, "window_end": 8},
            {"name": "Hartford, CT", "window_start": 3, "window_end": 10},
            {"name": "Albany, NY", "window_start": 5, "window_end": 15},
            {"name": "Pittsburgh, PA", "window_start": 8, "window_end": 14},
            {"name": "Buffalo, NY", "window_start": 10, "window_end": 20},
            # Midwest cluster (CRA 1 territory)
            {"name": "Detroit, MI", "window_start": 1, "window_end": 7},
            {"name": "Indianapolis, IN", "window_start": 3, "window_end": 11},
            {"name": "Milwaukee, WI", "window_start": 6, "window_end": 14},
            {"name": "Columbus, OH", "window_start": 8, "window_end": 16},
            {"name": "St. Louis, MO", "window_start": 12, "window_end": 20},
            # Southeast cluster (CRA 2 territory)
            {"name": "Charlotte, NC", "window_start": 1, "window_end": 5},
            {"name": "Nashville, TN", "window_start": 3, "window_end": 10},
            {"name": "Jacksonville, FL", "window_start": 5, "window_end": 15},
            {"name": "Miami, FL", "window_start": 10, "window_end": 20},
            {"name": "Richmond, VA", "window_start": 2, "window_end": 8},
        ],
        "max_days": 25,
    },

    # HARD: 10 CRAs, 50 sites nationwide, tight windows
    # Not all sites may be reachable — agent must prioritize
    # Cross-country travel takes multiple days
    # A greedy agent should score ~0.55-0.70
    "hard": {
        "task_id": "hard",
        "description": (
            "10 CRAs spread across the US must visit 50 clinical trial sites "
            "nationwide. Time windows are tight (3-8 days) and many overlap. "
            "Cross-country travel takes 2-4 days, making assignment critical. "
            "Not all sites may be reachable — the agent must prioritize high-value "
            "visits and minimize wasted travel. This simulates a real nationwide "
            "clinical trial monitoring scenario."
        ),
        "cras": [
            {"id": i, "home_city": _HARD_CRA_HOMES[i]}
            for i in range(10)
        ],
        "sites": _generate_sites(
            _available_sites(_HARD_CRA_HOMES),
            num_sites=50,
            day_range=(1, 25),
            window_size_range=(3, 8),
            seed=789,
            add_priority=True,
        ),
        "max_days": 40,
    },
}

ALL_TASK_IDS = list(TASKS.keys())


def get_task(task_id: str) -> dict:
    """Return task definition by ID."""
    return TASKS[task_id]
