"""CRA Scheduler Environment — core logic."""

import uuid
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import CRAAction, CRAObservation, CRAState
    from ..tasks import get_task
    from ..distances import get_distance, get_travel_days, MAX_DISTANCE
except ImportError:
    from models import CRAAction, CRAObservation, CRAState
    from tasks import get_task
    from distances import get_distance, get_travel_days, MAX_DISTANCE


class CRASchedulerEnvironment(Environment):
    """Environment where an agent schedules CRA site visits.

    Simulates multi-CRA clinical trial site visit scheduling with time windows.
    Based on the Vehicle Routing Problem with Time Windows (VRPTW).
    """

    def __init__(self, task_id: str = "easy"):
        super().__init__()
        self._task_id = task_id
        self._state = CRAState()
        self._task = None

    def get_metadata(self):
        from openenv.core.env_server.types import EnvironmentMetadata
        return EnvironmentMetadata(
            name="CRA Site Visit Scheduler",
            description=(
                "Multi-CRA clinical trial site visit scheduling with time windows. "
                "Agent assigns sites to CRAs and plans routes to minimize travel cost "
                "while meeting all visit deadlines. Based on VRPTW."
            ),
            version="1.0.0",
            author="Jitendra Singh Sankhwar",
            documentation_url="https://huggingface.co/spaces/JitendraSinghSankhwar/cra-scheduler-env",
        )
        self._cras = []
        self._unvisited = []
        self._visited = []
        self._total_cost = 0.0
        self._current_day = 1
        self._optimal_cost = None
        self._done = False

    def reset(self, seed=None, episode_id=None, **kwargs):
        task_id = kwargs.get("task_id", self._task_id)
        self._task_id = task_id
        self._task = get_task(task_id)
        self._current_day = 1
        self._total_cost = 0.0
        self._done = False

        # Initialize CRAs
        self._cras = []
        for cra in self._task["cras"]:
            self._cras.append({
                "id": cra["id"],
                "home_city": cra["home_city"],
                "current_city": cra["home_city"],
                "current_day": 1,
            })

        # Initialize sites
        self._unvisited = []
        for i, site in enumerate(self._task["sites"]):
            self._unvisited.append({
                "index": i,
                "name": site["name"],
                "window_start": site["window_start"],
                "window_end": site["window_end"],
            })
        self._visited = []

        # Compute optimal solution (skip for large tasks — too slow)
        num_sites = len(self._task["sites"])
        if num_sites <= 20:
            try:
                from .solver import solve
                result = solve(self._task)
                self._optimal_cost = result["total_cost"] if result else None
            except Exception:
                self._optimal_cost = None
        else:
            self._optimal_cost = None

        self._state = CRAState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_id=task_id,
            sites_remaining=len(self._unvisited),
            sites_visited=0,
            total_cost=0.0,
            current_day=1,
            optimal_cost=self._optimal_cost,
        )

        return self._make_observation(reward=0.0)

    def step(self, action: CRAAction, timeout_s=None, **kwargs):
        if self._done:
            return self._make_observation(reward=0.0)

        cra_id = action.cra_id
        site_index = action.site_index

        # Validate action
        if cra_id < 0 or cra_id >= len(self._cras):
            return self._make_observation(reward=-1.0, metadata={"error": "invalid cra_id"})
        if site_index < 0 or site_index >= len(self._unvisited):
            return self._make_observation(reward=-1.0, metadata={"error": "invalid site_index"})

        cra = self._cras[cra_id]
        site = self._unvisited[site_index]

        # Compute travel
        travel_cost = get_distance(cra["current_city"], site["name"])
        travel_days = get_travel_days(cra["current_city"], site["name"])
        arrival_day = cra["current_day"] + travel_days

        # Check time window
        reward = 0.0
        priority = site.get("priority", "normal")
        priority_multiplier = {"critical": 3.0, "high": 2.0, "normal": 1.0}.get(priority, 1.0)

        if arrival_day > site["window_end"]:
            # Missed window
            reward = -3.0 * priority_multiplier
            self._visited.append({
                "name": site["name"],
                "visited_by_cra": cra_id,
                "visited_on_day": None,
                "cost": travel_cost,
                "travel_days": travel_days,
                "status": "missed",
            })
            cra["current_day"] = arrival_day
            cra["current_city"] = site["name"]
        else:
            visit_day = max(arrival_day, site["window_start"])
            waited = visit_day > arrival_day
            reward = (1.0 if waited else 2.0) * priority_multiplier

            cra["current_day"] = visit_day
            cra["current_city"] = site["name"]

            self._visited.append({
                "name": site["name"],
                "visited_by_cra": cra_id,
                "visited_on_day": visit_day,
                "cost": travel_cost,
                "travel_days": travel_days,
                "status": "visited",
            })

        # Travel cost penalty (normalized)
        reward -= travel_cost / MAX_DISTANCE

        self._total_cost += travel_cost
        self._unvisited.pop(site_index)

        # Update current day (global = max across CRAs)
        self._current_day = max(c["current_day"] for c in self._cras)

        # Check if done
        if len(self._unvisited) == 0:
            # Add return-home costs
            for c in self._cras:
                return_cost = get_distance(c["current_city"], c["home_city"])
                self._total_cost += return_cost
            successful = sum(1 for v in self._visited if v["status"] == "visited")
            if successful == len(self._task["sites"]):
                reward += 5.0
            self._done = True
        else:
            # Check if any remaining site is impossible for all CRAs
            max_days = self._task.get("max_days", 60)
            all_impossible = True
            for s in self._unvisited:
                for c in self._cras:
                    days_needed = get_travel_days(c["current_city"], s["name"])
                    if c["current_day"] + days_needed <= s["window_end"] and c["current_day"] + days_needed <= max_days:
                        all_impossible = False
                        break
                if not all_impossible:
                    break
            if all_impossible or self._current_day > max_days:
                reward -= 5.0
                self._done = True

        self._state = CRAState(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count + 1,
            task_id=self._task_id,
            sites_remaining=len(self._unvisited),
            sites_visited=len(self._visited),
            total_cost=self._total_cost,
            current_day=self._current_day,
            optimal_cost=self._optimal_cost,
        )

        return self._make_observation(reward=reward)

    @property
    def state(self):
        return self._state

    def _make_observation(self, reward=0.0, metadata=None):
        unvisited_with_distances = []
        for site in self._unvisited:
            distances = {}
            travel_days = {}
            for cra in self._cras:
                distances[f"cra_{cra['id']}"] = get_distance(cra["current_city"], site["name"])
                travel_days[f"cra_{cra['id']}"] = get_travel_days(cra["current_city"], site["name"])
            entry = {**site, "distances": distances, "travel_days": travel_days}
            unvisited_with_distances.append(entry)

        missed = sum(1 for v in self._visited if v.get("status") == "missed")

        return CRAObservation(
            current_day=self._current_day,
            cras=[{
                "id": c["id"],
                "home_city": c["home_city"],
                "current_city": c["current_city"],
                "current_day": c["current_day"],
            } for c in self._cras],
            unvisited_sites=unvisited_with_distances,
            visited_sites=self._visited,
            total_cost=self._total_cost,
            task_id=self._task_id,
            done=self._done,
            reward=reward,
            sites_missed=missed,
            metadata=metadata or {},
        )
