"""Baseline inference script — runs LLM against all 3 CRA scheduling tasks."""

import asyncio
import json
import os
import sys
from typing import List, Optional

from openai import OpenAI

from cra_scheduler_env import CRAAction, CRASchedulerEnv
from cra_scheduler_env.grader import grade
from cra_scheduler_env.tasks import ALL_TASK_IDS

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.getenv("CRA_ENV_URL", "http://localhost:8000")
BENCHMARK = "cra_scheduler_env"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def format_prompt(obs) -> str:
    lines = [
        "You are a travel planner for Clinical Research Associates (CRAs).",
        f"Current day: {obs.current_day}",
        "",
        "CRAs:",
    ]
    for cra in obs.cras:
        lines.append(f"  CRA {cra['id']}: at {cra['current_city']} (home: {cra['home_city']}, day: {cra.get('current_day', '?')})")
    lines.append("")
    lines.append("Unvisited sites:")
    for i, site in enumerate(obs.unvisited_sites):
        dists = ", ".join(f"CRA {k}: {v} mi" for k, v in site.get("distances", {}).items())
        lines.append(f"  {i}: {site['name']} (window: day {site['window_start']}-{site['window_end']}) [{dists}]")
    lines.append("")
    lines.append(f"Total cost so far: {obs.total_cost} miles")
    lines.append("")
    lines.append("Pick the best (cra_id, site_index) to minimize total travel cost while meeting all time windows.")
    lines.append('Respond with ONLY JSON: {"cra_id": N, "site_index": N}')
    return "\n".join(lines)


def parse_response(text: str) -> dict:
    text = text.strip()
    for start in range(len(text)):
        if text[start] == "{":
            for end in range(len(text), start, -1):
                if text[end - 1] == "}":
                    try:
                        return json.loads(text[start:end])
                    except json.JSONDecodeError:
                        continue
    return {"cra_id": 0, "site_index": 0}


async def run_task(client_llm: OpenAI, env, task_id: str) -> dict:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    sites_visited = 0
    cost = 0.0

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation
        total_sites = len(obs.unvisited_sites) + len(obs.visited_sites)

        while not result.done:
            if not obs.unvisited_sites:
                break

            try:
                prompt = format_prompt(obs)
                response = client_llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.0,
                )
                parsed = parse_response(response.choices[0].message.content)
            except Exception as e:
                print(f"[DEBUG] LLM error: {e}", flush=True)
                parsed = {"cra_id": 0, "site_index": 0}

            action = CRAAction(
                cra_id=parsed.get("cra_id", 0),
                site_index=parsed.get("site_index", 0),
            )
            action_str = f"CRAAction(cra_id={action.cra_id},site_index={action.site_index})"

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            steps_taken += 1
            rewards.append(reward)

            log_step(step=steps_taken, action=action_str, reward=reward, done=result.done, error=None)

        sites_visited = sum(1 for v in obs.visited_sites if v.get("status") == "visited")
        cost = obs.total_cost
        state = await env.state()
        optimal = state.optimal_cost or obs.total_cost
        score = grade(sites_visited, total_sites, obs.total_cost, optimal)
        score = min(max(score, 0.001), 0.999)
        success = score > 0.0

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "score": score, "cost": cost, "sites_visited": sites_visited}


async def main():
    if not API_KEY:
        print("Error: set HF_TOKEN or API_KEY environment variable", flush=True)
        sys.exit(1)

    client_llm = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    print("CRA Scheduler — Baseline Inference", flush=True)
    print("=" * 50, flush=True)

    async with CRASchedulerEnv(base_url=ENV_URL) as env:
        results = []
        for task_id in ALL_TASK_IDS:
            print(f"\nRunning task: {task_id}...", flush=True)
            result = await run_task(client_llm, env, task_id)
            results.append(result)
            print(f"  Score: {result['score']:.4f} | Cost: {result['cost']:.0f} mi", flush=True)

    print("\n" + "=" * 50, flush=True)
    print("Summary:", flush=True)
    for r in results:
        print(f"  {r['task_id']:8s}: {r['score']:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
