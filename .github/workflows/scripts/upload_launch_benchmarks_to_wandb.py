"""
Upload launch-overhead benchmark results to W&B.

Reads a pipe-delimited results file produced by test_launch_overhead.py:
    scenario=many_structs_cached 	| launches_per_sec=123456.7

Uploads each scenario's launches_per_sec as a W&B summary metric keyed as
"launches_per_sec-scenario=<name>" so that alarm.py can fetch them.
"""

import argparse
import subprocess
import uuid

import wandb


def get_git_revision() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], encoding="utf-8").strip()
        branches = subprocess.check_output(
            ["git", "branch", "-r", "--contains", commit],
            encoding="utf-8",
        ).strip()
        branch = "UNKNOWN"
        for line in branches.splitlines():
            line = line.strip().removeprefix("* ")
            if "->" not in line:
                branch = line
                break
        return f"{commit}@{branch}"
    except subprocess.CalledProcessError:
        return f"{uuid.uuid4().hex}@UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-prefix", default="launch")
    args = parser.parse_args()

    revision = get_git_revision()
    print(f"Uploading to W&B project '{args.project}' for revision: {revision}")

    run = wandb.init(
        project=args.project,
        name=f"{args.run_prefix}-{revision[:12]}",
        config={"revision": revision},
        settings=wandb.Settings(x_disable_stats=True, console="off"),
    )

    count = 0
    with open(args.in_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            params = {}
            for part in line.split(" \t| "):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k.strip()] = v.strip()
            scenario = params.get("scenario")
            lps = params.get("launches_per_sec")
            if scenario and lps:
                key = f"launches_per_sec-scenario={scenario}"
                run.log({key: float(lps)})
                print(f"  {key}: {lps}")
                count += 1

    run.finish()
    print(f"Uploaded {count} results")


if __name__ == "__main__":
    main()
