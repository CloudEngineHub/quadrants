"""
Compare launch-overhead benchmark results from the current PR against historical data on W&B.

Reads a pipe-delimited results file produced by test_launch_overhead.py (via --speed-test-filepath), fetches baseline
data from the quadrants-benchmarks W&B project, and checks for regressions. Outputs a markdown report and exits with a
special code when regressions are found.

Pipe-delimited format (one line per scenario):
    scenario=many_structs_cached 	| launches_per_sec=123456.7

This is intentionally much simpler than the Genesis alarm.py — we have a single metric (launches_per_sec) and a handful
of scenarios, with no memory or compile-time tracking.
"""

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import wandb


def parse_results_file(path: Path) -> dict[str, float]:
    """Parse pipe-delimited results file into {scenario: launches_per_sec}."""
    results: dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        params: dict[str, str] = {}
        for part in line.split(" \t| "):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k.strip()] = v.strip()
        scenario = params.get("scenario")
        lps = params.get("launches_per_sec")
        if scenario and lps:
            results[scenario] = float(lps)
    return results


def fetch_wandb_baselines(
    entity: str,
    project: str,
    scenarios: set[str],
    max_revisions: int,
    max_fetch: int,
) -> dict[str, list[float]]:
    """Fetch historical launches_per_sec values from W&B, keyed by scenario."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", order="-created_at")

    baselines: dict[str, list[float]] = defaultdict(list)
    commit_hashes: set[str] = set()

    for run in runs:
        if len(commit_hashes) >= max_fetch:
            break
        complete_scenarios = sum(1 for s in scenarios if len(baselines.get(s, [])) >= max_revisions)
        if complete_scenarios == len(scenarios):
            break

        try:
            config, summary = run.config, run.summary
        except Exception:
            continue

        if isinstance(config, str):
            config = {k: v["value"] for k, v in json.loads(config).items() if not k.startswith("_")}

        revision = config.get("revision", "")
        if "@" not in revision:
            continue
        commit_hash, branch = revision.split("@", 1)
        if not branch.startswith("Genesis-Embodied-AI/"):
            continue
        if run.state != "finished":
            continue

        commit_hashes.add(commit_hash)

        for key, val in (summary._json_dict if hasattr(summary, "_json_dict") else summary).items():
            if key.startswith("_"):
                continue
            # W&B keys are like "launches_per_sec-scenario=many_structs_cached"
            if not key.startswith("launches_per_sec-"):
                continue
            scenario_part = key[len("launches_per_sec-") :]
            # scenario_part is "scenario=many_structs_cached"
            if "=" in scenario_part:
                scenario_name = scenario_part.split("=", 1)[1]
            else:
                scenario_name = scenario_part
            if scenario_name in scenarios and len(baselines[scenario_name]) < max_revisions:
                try:
                    baselines[scenario_name].append(float(val))
                except (TypeError, ValueError):
                    pass

    return dict(baselines)


def build_markdown(
    current: dict[str, float],
    baselines: dict[str, list[float]],
    tolerance_pct: float,
    min_revisions: int,
) -> tuple[str, bool, bool]:
    """Build a markdown comparison table. Returns (markdown, has_regressions, has_alerts)."""
    lines = [
        f"Threshold: launches/sec ± {tolerance_pct:.0f}%",
        "",
        "### Launch Overhead",
        "",
        "| Status | Scenario | Current launches/s | Baseline launches/s [last (mean ± std)] | Δ |",
        "|:------:|:---------|-------------------:|----------------------------------------:|--:|",
    ]

    reg_found = False
    alert_found = False

    for scenario in sorted(current.keys()):
        cur = current[scenario]
        hist = baselines.get(scenario, [])

        if not hist:
            lines.append(f"| ℹ️ | {scenario} | {cur:,.0f} | --- | --- |")
            continue

        last = hist[0]
        delta_pct = (cur - last) / last * 100.0

        if len(hist) >= min_revisions:
            mean = statistics.fmean(hist)
            ci95 = statistics.stdev(hist) / math.sqrt(len(hist)) * 1.96 if len(hist) > 1 else math.nan
            stats_repr = f"{last:,.0f} ({mean:,.0f} ± {ci95:,.0f})"

            if delta_pct < -tolerance_pct:
                icon = "🔴"
                delta_repr = f"**{delta_pct:+.1f}%**"
                reg_found = True
            elif delta_pct > tolerance_pct:
                icon = "⚠️"
                delta_repr = f"**{delta_pct:+.1f}%**"
                alert_found = True
            else:
                icon = "✅"
                delta_repr = f"{delta_pct:+.1f}%"
        else:
            stats_repr = f"{last:,.0f}"
            icon = "ℹ️"
            delta_repr = f"{delta_pct:+.1f}%"

        lines.append(f"| {icon} | {scenario} | {cur:,.0f} | {stats_repr} | {delta_repr} |")

    n_baselines = len(set().union(*(set(range(len(v))) for v in baselines.values()))) if baselines else 0
    lines.append("")
    lines.append(f"**Baselines considered:** {n_baselines} commits")

    return "\n".join(lines), reg_found, alert_found


def main():
    parser = argparse.ArgumentParser(description="Quadrants launch-overhead regression alarm")
    parser.add_argument("--artifacts-dir", required=True, help="Directory containing speed_test*.txt artifacts")
    parser.add_argument("--max-valid-revisions", type=int, default=5)
    parser.add_argument("--max-fetch-revisions", type=int, default=30)
    parser.add_argument("--tolerance-pct", type=float, default=10.0)
    parser.add_argument("--check-body-path", required=True, help="Path to write markdown output")
    parser.add_argument("--exit-code-regression", type=int, default=42)
    parser.add_argument("--exit-code-alert", type=int, default=43)
    args = parser.parse_args()

    entity = "genesis-ai-company"
    project = "quadrants-benchmarks"

    artifacts_dir = Path(args.artifacts_dir).resolve()
    result_files = list(artifacts_dir.rglob("speed_test*.txt"))
    if not result_files:
        print(f"No speed_test*.txt files found in {artifacts_dir}", file=sys.stderr)
        sys.exit(1)

    current: dict[str, float] = {}
    for f in result_files:
        current.update(parse_results_file(f))

    if not current:
        print("No benchmark results parsed from artifact files", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(current)} scenarios from {len(result_files)} artifact file(s)")

    baselines = fetch_wandb_baselines(
        entity=entity,
        project=project,
        scenarios=set(current.keys()),
        max_revisions=args.max_valid_revisions,
        max_fetch=args.max_fetch_revisions,
    )

    md, reg_found, alert_found = build_markdown(
        current=current,
        baselines=baselines,
        tolerance_pct=args.tolerance_pct,
        min_revisions=args.max_valid_revisions,
    )

    Path(args.check_body_path).write_text(md + "\n", encoding="utf-8")
    print(md)

    if reg_found:
        sys.exit(args.exit_code_regression)
    if alert_found:
        sys.exit(args.exit_code_alert)
    sys.exit(0)


if __name__ == "__main__":
    main()
