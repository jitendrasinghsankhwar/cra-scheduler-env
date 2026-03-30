"""FastAPI application for CRA Scheduler Environment."""

from typing import Dict, Optional
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from openenv.core.env_server.http_server import create_app
    from ..models import CRAAction, CRAObservation
    from ..tasks import TASKS, ALL_TASK_IDS
    from ..grader import grade
    from .environment import CRASchedulerEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from models import CRAAction, CRAObservation
    from tasks import TASKS, ALL_TASK_IDS
    from grader import grade
    from server.environment import CRASchedulerEnvironment


# --- HTTP Session Store ---
_sessions: Dict[str, CRASchedulerEnvironment] = {}


class SessionResetRequest(BaseModel):
    task_id: str = "easy"


class SessionStepRequest(BaseModel):
    session_id: str
    cra_id: int
    site_index: int


def _serialize_obs(obs: CRAObservation) -> dict:
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


# --- Standard OpenEnv app ---
app = create_app(
    CRASchedulerEnvironment,
    CRAAction,
    CRAObservation,
    env_name="cra_scheduler_env",
    max_concurrent_envs=1,
)


# --- Custom endpoints ---

@app.get("/tasks")
def list_tasks():
    """Return all 3 tasks with descriptions and action schema."""
    tasks_info = []
    for tid in ALL_TASK_IDS:
        t = TASKS[tid]
        tasks_info.append({
            "task_id": t["task_id"],
            "description": t["description"],
            "num_cras": len(t["cras"]),
            "num_sites": len(t["sites"]),
            "action_schema": CRAAction.model_json_schema(),
        })
    return JSONResponse(content={"tasks": tasks_info})


@app.get("/grader")
def get_grader_score(
    sites_visited: int = 0,
    total_sites: int = 0,
    agent_cost: float = 0.0,
    optimal_cost: float = 0.0,
):
    """Return grader score for given metrics."""
    score = grade(sites_visited, total_sites, agent_cost, optimal_cost)
    return JSONResponse(content={"score": score})


@app.post("/baseline")
def run_baseline():
    """Run greedy nearest-neighbor baseline and return scores for all tasks."""
    results = []
    for task_id in ALL_TASK_IDS:
        env = CRASchedulerEnvironment(task_id=task_id)
        obs = env.reset()
        total_sites = len(obs.unvisited_sites)

        while not obs.done and len(obs.unvisited_sites) > 0:
            best_cra, best_site, best_dist = 0, 0, float("inf")
            for si, site in enumerate(obs.unvisited_sites):
                for cra in obs.cras:
                    d = site["distances"].get(f"cra_{cra['id']}", float("inf"))
                    if d < best_dist:
                        best_dist = d
                        best_cra = cra["id"]
                        best_site = si
            obs = env.step(CRAAction(cra_id=best_cra, site_index=best_site))

        visited = sum(1 for v in obs.visited_sites if v.get("status") == "visited")
        optimal = env.state.optimal_cost or obs.total_cost
        score = grade(visited, total_sites, obs.total_cost, optimal)
        results.append({
            "task_id": task_id,
            "score": score,
            "cost": obs.total_cost,
            "sites_visited": visited,
            "total_sites": total_sites,
            "optimal_cost": optimal,
        })

    return JSONResponse(content={"results": results})


# --- HTTP Session Endpoints (stateful over HTTP) ---

@app.post("/session/reset")
def session_reset(req: SessionResetRequest):
    """Start a new stateful HTTP session. Returns session_id."""
    env = CRASchedulerEnvironment(task_id=req.task_id)
    obs = env.reset()
    session_id = env.state.episode_id
    _sessions[session_id] = env
    resp = _serialize_obs(obs)
    resp["session_id"] = session_id
    return JSONResponse(content=resp)


@app.post("/session/step")
def session_step(req: SessionStepRequest):
    """Take a step in an existing HTTP session."""
    env = _sessions.get(req.session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found. Call /session/reset first.")
    obs = env.step(CRAAction(cra_id=req.cra_id, site_index=req.site_index))
    resp = _serialize_obs(obs)
    resp["session_id"] = req.session_id
    if obs.done:
        # Include grader score
        visited = sum(1 for v in obs.visited_sites if v.get("status") == "visited")
        total = len(obs.visited_sites)
        optimal = env.state.optimal_cost or obs.total_cost
        resp["grader_score"] = grade(visited, total, obs.total_cost, optimal)
        del _sessions[req.session_id]  # cleanup
    return JSONResponse(content=resp)


@app.get("/session/state")
def session_state(session_id: str):
    """Get state of an existing HTTP session."""
    env = _sessions.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found.")
    return JSONResponse(content=env.state.model_dump())


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
