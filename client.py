"""CRA Scheduler Environment Client."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import CRAAction, CRAObservation, CRAState


class CRASchedulerEnv(EnvClient[CRAAction, CRAObservation, CRAState]):
    """Client for CRA Scheduler Environment."""

    def _step_payload(self, action: CRAAction) -> Dict[str, Any]:
        return {
            "cra_id": action.cra_id,
            "site_index": action.site_index,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[CRAObservation]:
        obs_data = payload.get("observation", {})
        observation = CRAObservation(
            current_day=obs_data.get("current_day", 1),
            cras=obs_data.get("cras", []),
            unvisited_sites=obs_data.get("unvisited_sites", []),
            visited_sites=obs_data.get("visited_sites", []),
            total_cost=obs_data.get("total_cost", 0.0),
            task_id=obs_data.get("task_id", "easy"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CRAState:
        return CRAState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "easy"),
            sites_remaining=payload.get("sites_remaining", 0),
            sites_visited=payload.get("sites_visited", 0),
            total_cost=payload.get("total_cost", 0.0),
            current_day=payload.get("current_day", 1),
            optimal_cost=payload.get("optimal_cost"),
        )
