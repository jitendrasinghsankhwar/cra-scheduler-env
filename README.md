---

# CRA Site Visit Scheduler Environment

An OpenEnv reinforcement learning environment where an AI agent learns to optimally schedule Clinical Research Associate (CRA) visits to clinical trial sites.

## Motivation

In clinical trials, CRAs monitor trial sites through periodic visits. Each visit has a time window — a date range within which it must occur. CRAs manage multiple sites across different cities. Planning manually is slow and suboptimal, leading to unnecessary travel costs and missed visit windows.

This environment models the **Multi-Vehicle Routing Problem with Time Windows (VRPTW)** — a real operations research problem used in the pharmaceutical industry.

## Tasks

| Task | CRAs | Sites | Windows | Description |
|------|------|-------|---------|-------------|
| `easy` | 1 | 4 | 15-25 days | 1 CRA, 4 nearby Northeast sites, wide windows. Any reasonable route works. |
| `medium` | 3 | 15 | 4-14 days | 3 CRAs across East/Midwest, mixed windows. Smart CRA assignment required. |
| `hard` | 10 | 50 | 3-8 days | 10 CRAs nationwide, tight windows with site priorities. Not all sites reachable. |

The hard task includes **site priority levels** (critical/high/normal) — critical sites give 3x reward, missed critical sites give 3x penalty.

## Action Space

```python
CRAAction(cra_id=0, site_index=1)  # Send CRA 0 to unvisited site at index 1
```

| Field | Type | Description |
|-------|------|-------------|
| `cra_id` | int | Which CRA to send (0-indexed) |
| `site_index` | int | Which unvisited site to visit (0-indexed into unvisited list) |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `current_day` | int | Current day in the planning period |
| `cras` | list[dict] | CRA states: id, home_city, current_city, current_day |
| `unvisited_sites` | list[dict] | Sites not yet visited: name, window_start, window_end, distances, travel_days, priority |
| `visited_sites` | list[dict] | Sites already visited: name, visited_by_cra, visited_on_day, cost, travel_days, status |
| `total_cost` | float | Total travel cost so far (miles) |
| `sites_missed` | int | Number of sites whose windows have closed |
| `task_id` | str | Current task ID |
| `done` | bool | Whether episode is finished |
| `reward` | float | Reward for last action |

## Reward Design

| Event | Reward | Notes |
|-------|--------|-------|
| Visit within window | +2.0 × priority | critical=3x, high=2x, normal=1x |
| Visit but waited (early arrival) | +1.0 × priority | Arrived before window opened |
| Missed window | -3.0 × priority | Window closed before visit |
| Travel cost per step | -(distance / max_distance) | Normalized penalty |
| All sites completed | +5.0 | Bonus for full completion |
| Episode impossible | -5.0 | No remaining sites reachable |

## Grading

Each task scored 0.0 to 1.0:
```
completion = sites_visited / total_sites
efficiency = min(1.0, optimal_cost / agent_cost)
score = completion × 0.5 + efficiency × 0.5
```

OR-Tools computes the optimal solution as benchmark (server-side only).

## Baseline Scores

Greedy nearest-neighbor baseline:

| Task | Score | Visited | Cost (mi) |
|------|-------|---------|-----------|
| easy | 1.0000 | 4/4 | 279 |
| medium | 0.7954 | 11/15 | 6,699 |
| hard | 0.7900 | 29/50 | 15,273 |

## Setup

```bash
# Run server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run baseline
export OPENAI_API_KEY=your_key
python baseline.py
```

## Docker

```bash
docker build -t cra-scheduler-env:latest -f server/Dockerfile .
docker run -p 8000:8000 cra-scheduler-env:latest
```

## Usage (WebSocket — recommended)

```python
import asyncio, json
from websockets.asyncio.client import connect

async def play():
    async with connect("wss://JitendraSinghSankhwar-cra-scheduler-env.hf.space/ws") as ws:
        await ws.send(json.dumps({"type": "reset", "data": {"task_id": "easy"}}))
        r = json.loads(await ws.recv())
        obs = r["data"]["observation"]

        while not r["data"].get("done", False):
            sites = obs["unvisited_sites"]
            if not sites:
                break
            await ws.send(json.dumps({"type": "step", "data": {"cra_id": 0, "site_index": 0}}))
            r = json.loads(await ws.recv())
            obs = r["data"]["observation"]

asyncio.run(play())
```

## Usage (HTTP Session)

```bash
# Start session
curl -X POST /session/reset -d '{"task_id":"easy"}'
# Returns session_id

# Take steps
curl -X POST /session/step -d '{"session_id":"<id>","cra_id":0,"site_index":0}'
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/tasks` | GET | List all 3 tasks with action schema |
| `/schema` | GET | Action/observation/state JSON schemas |
| `/baseline` | POST | Run greedy baseline, return scores |
| `/grader` | GET | Compute grader score from metrics |
| `/reset` | POST | Reset environment (stateless) |
| `/step` | POST | Execute action (stateless) |
| `/state` | GET | Current state |
| `/ws` | WS | WebSocket for stateful sessions |
| `/session/reset` | POST | Start stateful HTTP session |
| `/session/step` | POST | Step in HTTP session |
| `/session/state` | GET | Get HTTP session state |

## Author

Jitendra Singh Sankhwar — OpenEnv Hackathon 2026
