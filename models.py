"""Data models for CRA Scheduler Environment.

Defines the typed Pydantic models for the OpenEnv interface:
- CRAAction: what the agent sends each step (which CRA visits which site)
- CRAObservation: what the agent sees (CRA positions, sites, costs)
- CRAState: episode bookkeeping (step count, scores, optimal cost)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class CRAAction(Action):
    """Agent picks which CRA visits which site next.

    Each step, the agent selects one (cra_id, site_index) pair.
    The CRA travels to that site, incurring travel cost and time.
    """

    cra_id: int = Field(description="Which CRA to send (0-indexed)")
    site_index: int = Field(description="Which unvisited site to visit (0-indexed into unvisited list)")


class CRAObservation(Observation):
    """What the agent sees after each step.

    Contains full state needed for decision-making: CRA positions,
    remaining sites with distances/windows, and cost so far.
    """

    current_day: int = Field(default=1)
    cras: List[Dict[str, Any]] = Field(default_factory=list, description="CRA states: id, home_city, current_city")
    unvisited_sites: List[Dict[str, Any]] = Field(default_factory=list, description="Sites not yet visited")
    visited_sites: List[Dict[str, Any]] = Field(default_factory=list, description="Sites already visited")
    total_cost: float = Field(default=0.0, description="Total travel cost in miles")
    task_id: str = Field(default="easy")
    sites_missed: int = Field(default=0, description="Number of sites whose windows have closed")


class CRAState(State):
    """Episode bookkeeping — tracks progress and optimal benchmark."""

    task_id: str = Field(default="easy")
    sites_remaining: int = Field(default=0)
    sites_visited: int = Field(default=0)
    total_cost: float = Field(default=0.0)
    current_day: int = Field(default=1)
    optimal_cost: Optional[float] = Field(default=None, description="OR-Tools optimal cost for comparison")
