# 🏥 CRA Site Visit Scheduler

An OpenEnv RL environment for scheduling Clinical Research Associate (CRA) visits to clinical trial sites — a real operations research problem from the pharmaceutical industry.

**Live:** [HF Space](https://huggingface.co/spaces/JitendraSinghSankhwar/cra-visit-scheduler) · **Docs:** [API Playground](https://JitendraSinghSankhwar-cra-visit-scheduler.hf.space/docs)

## The Problem

CRAs monitor clinical trial sites through periodic visits. Each site has a **time window** — a date range within which the visit must happen. CRAs are based in different cities, and travel between sites costs time and money.

The agent must:
1. **Assign** sites to CRAs (who covers what)
2. **Route** visits to minimize total travel cost
3. **Meet deadlines** — visit each site within its time window

This is the [Vehicle Routing Problem with Time Windows (VRPTW)](https://en.wikipedia.org/wiki/Vehicle_routing_problem) — an NP-hard optimization problem used daily in pharma logistics.

## Tasks

| Task | CRAs | Sites | Window Width | What Makes It Hard |
|------|------|-------|-------------|--------------------|
| `easy` | 1 | 4 | 15–25 days | Just find a good route order |
| `medium` | 3 | 15 | 4–14 days | Must assign sites to the right CRA |
| `hard` | 10 | 50 | 3–8 days | Not all sites reachable; must prioritize |

## How It Works

Each step, the agent picks one `(cra_id, site_index)` pair:

```python
CRAAction(cra_id=0, site_index=2)  # Send CRA 0 to the 3rd unvisited site
```

The environment returns an observation with CRA positions, unvisited sites with distances and time windows, visited sites, and total cost. Rewards are given per step:

| Event | Reward |
|-------|--------|
| Visit within window | +2.0 × priority |
| Early arrival (waited) | +1.0 × priority |
| Missed window | −3.0 × priority |
| Travel cost | −(distance / max_distance) |
| All sites done | +5.0 bonus |

## Grading

```
score = 0.5 × (sites_visited / total_sites) + 0.5 × min(1, optimal_cost / agent_cost)
```

Optimal cost is computed server-side using [OR-Tools](https://developers.google.com/optimization).

## Quick Start

```python
import asyncio
from cra_scheduler_env import CRAAction, CRASchedulerEnv

async def main():
    async with CRASchedulerEnv(base_url="https://JitendraSinghSankhwar-cra-visit-scheduler.hf.space") as env:
        result = await env.reset(task_id="easy")
        while not result.done:
            obs = result.observation
            if not obs.unvisited_sites:
                break
            result = await env.step(CRAAction(cra_id=0, site_index=0))
        print(f"Cost: {result.observation.total_cost} miles")

asyncio.run(main())
```

## Run Locally

```bash
# Docker
docker build -t cra-scheduler-env .
docker run -p 8000:8000 cra-scheduler-env

# Or directly
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
├── models.py              # Action, Observation, State (Pydantic)
├── tasks.py               # 3 task definitions (easy/medium/hard)
├── distances.py           # 100 US cities with haversine distances
├── grader.py              # Scoring: completion + efficiency
├── client.py              # WebSocket client (EnvClient subclass)
├── inference.py           # LLM baseline using OpenAI API
├── server/
│   ├── environment.py     # Core step/reset/state logic
│   ├── solver.py          # OR-Tools optimal benchmark
│   └── app.py             # FastAPI + custom endpoints
├── Dockerfile
├── openenv.yaml
└── pyproject.toml
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /tasks` | List tasks with action schema |
| `POST /baseline` | Run greedy baseline, return scores |
| `GET /grader` | Compute score from metrics |
| `WS /ws` | WebSocket for stateful sessions |
| `POST /session/reset` | Start HTTP session |
| `POST /session/step` | Step in HTTP session |

## Author

Jitendra Singh Sankhwar — [OpenEnv Hackathon 2026](https://meta-pytorch.org/OpenEnv/)
