"""Baseline inference script — runs OpenAI LLM against all 3 tasks."""

import json
import os
import sys

from openai import OpenAI

from cra_scheduler_env import CRAAction, CRASchedulerEnv
from cra_scheduler_env.grader import grade
from cra_scheduler_env.tasks import ALL_TASK_IDS

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ENV_URL = os.getenv("CRA_ENV_URL", "http://localhost:8000")


def format_prompt(obs) -> str:
    """Format observation into a prompt for the LLM."""
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
    lines.append("Respond with ONLY JSON: {\"cra_id\": N, \"site_index\": N}")

    return "\n".join(lines)


def parse_response(text: str) -> dict:
    """Extract cra_id and site_index from LLM response."""
    text = text.strip()
    # Try to find JSON in the response
    for start in range(len(text)):
        if text[start] == "{":
            for end in range(len(text), start, -1):
                if text[end - 1] == "}":
                    try:
                        return json.loads(text[start:end])
                    except json.JSONDecodeError:
                        continue
    return {"cra_id": 0, "site_index": 0}


def run_task(client_llm: OpenAI, env: CRASchedulerEnv, task_id: str) -> dict:
    """Run one task, return grader score."""
    result = env.reset(task_id=task_id)
    obs = result.observation
    total_sites = len(obs.unvisited_sites) + len(obs.visited_sites)

    while not result.done:
        prompt = format_prompt(obs)
        response = client_llm.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
        )
        parsed = parse_response(response.choices[0].message.content)
        action = CRAAction(
            cra_id=parsed.get("cra_id", 0),
            site_index=parsed.get("site_index", 0),
        )
        result = env.step(action)
        obs = result.observation

    sites_visited = sum(1 for v in obs.visited_sites if v.get("status") == "visited")
    state = env.state()
    optimal = state.optimal_cost or obs.total_cost
    score = grade(sites_visited, total_sites, obs.total_cost, optimal)

    return {"task_id": task_id, "score": score, "cost": obs.total_cost, "sites_visited": sites_visited}


def main():
    if not API_KEY:
        print("Error: set OPENAI_API_KEY environment variable")
        sys.exit(1)

    client_llm = OpenAI(api_key=API_KEY)
    env = CRASchedulerEnv(base_url=ENV_URL)

    print("CRA Scheduler — Baseline Inference")
    print("=" * 50)

    results = []
    for task_id in ALL_TASK_IDS:
        print(f"\nRunning task: {task_id}...")
        result = run_task(client_llm, env, task_id)
        results.append(result)
        print(f"  Score: {result['score']:.4f} | Cost: {result['cost']:.0f} mi | Sites: {result['sites_visited']}")

    print("\n" + "=" * 50)
    print("Summary:")
    for r in results:
        print(f"  {r['task_id']:8s}: {r['score']:.4f}")

    env.close()


if __name__ == "__main__":
    main()
