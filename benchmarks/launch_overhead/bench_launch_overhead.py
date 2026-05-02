#!/usr/bin/env python3
"""
Quadrants kernel-launch overhead benchmarks.

These measure the Python-side cost of launching kernels with various parameter patterns,
independent of kernel computation. The kernels themselves do trivial work — we only care
about the launch path (argument processing, caching, struct traversal).

Scenarios modeled after Genesis rigid-body simulation patterns:
- Many frozen-dataclass struct parameters (mimics forward_kinematics, constraint_solver)
- Mix of cacheable (struct) and uncacheable (torch.Tensor) args (mimics set_dofs_position)
- Template annotations vs struct annotations (before/after always-fastcache migration)
"""
import argparse
import dataclasses
import json
import time

import torch

import quadrants as qd

N_WARMUP = 50
N_STEPS = 2000
N_TRIALS = 5


# ---------------------------------------------------------------------------
# Struct definitions — mimics Genesis array_class patterns (10-15 fields each)
# ---------------------------------------------------------------------------


def _make_struct_class(name, n_fields):
    """Dynamically create a frozen dataclass with n_fields, each typed as qd.Tensor."""
    fields = [(f"f{i}", qd.Tensor, dataclasses.field(default=None)) for i in range(n_fields)]
    return dataclasses.make_dataclass(name, fields, frozen=True)


StructA = _make_struct_class("StructA", 10)
StructB = _make_struct_class("StructB", 12)
StructC = _make_struct_class("StructC", 8)
StructD = _make_struct_class("StructD", 14)
StructE = _make_struct_class("StructE", 6)


# ---------------------------------------------------------------------------
# Kernel definitions (created lazily after qd.init)
# ---------------------------------------------------------------------------

_kernels_initialized = False
out_field = None
kernel_many_structs = None
kernel_structs_plus_tensor = None
kernel_template_annotations = None
kernel_template_plus_tensor = None


def init_kernels():
    global _kernels_initialized, out_field
    global kernel_many_structs, kernel_structs_plus_tensor
    global kernel_template_annotations, kernel_template_plus_tensor

    if _kernels_initialized:
        return
    _kernels_initialized = True

    out_field = qd.field(qd.i32, shape=())

    @qd.kernel
    def _kernel_many_structs(s1: StructA, s2: StructB, s3: StructC, s4: StructD, s5: StructE):
        out_field[None] = 1

    @qd.kernel
    def _kernel_structs_plus_tensor(
        s1: StructA, s2: StructB, s3: StructC, s4: StructD, s5: StructE, t: qd.types.ndarray()
    ):
        out_field[None] = 1

    @qd.kernel
    def _kernel_template_annotations(
        s1: qd.template(), s2: qd.template(), s3: qd.template(), s4: qd.template(), s5: qd.template()
    ):
        out_field[None] = 1

    @qd.kernel
    def _kernel_template_plus_tensor(
        s1: qd.template(), s2: qd.template(), s3: qd.template(), s4: qd.template(), s5: qd.template(),
        t: qd.types.ndarray(),
    ):
        out_field[None] = 1

    kernel_many_structs = _kernel_many_structs
    kernel_structs_plus_tensor = _kernel_structs_plus_tensor
    kernel_template_annotations = _kernel_template_annotations
    kernel_template_plus_tensor = _kernel_template_plus_tensor


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def make_field_struct(cls):
    """Create a frozen dataclass instance where all fields are qd.field (zero-slot)."""
    fields = dataclasses.fields(cls)
    kwargs = {f.name: qd.field(qd.f32, shape=(4,)) for f in fields}
    return cls(**kwargs)


def run_benchmark(name, step_fn, n_warmup=N_WARMUP, n_steps=N_STEPS, n_trials=N_TRIALS):
    """Run a benchmark, return median launches/sec."""
    for _ in range(n_warmup):
        step_fn()

    results = []
    for trial in range(n_trials):
        t0 = time.perf_counter()
        for _ in range(n_steps):
            step_fn()
        elapsed = time.perf_counter() - t0
        launches_per_sec = n_steps / elapsed
        results.append(launches_per_sec)

    median = sorted(results)[len(results) // 2]
    return {"name": name, "median_launches_per_sec": median, "all_results": results}


def run_all():
    """Run all benchmarks, return list of result dicts."""
    init_kernels()
    sa = make_field_struct(StructA)
    sb = make_field_struct(StructB)
    sc = make_field_struct(StructC)
    sd = make_field_struct(StructD)
    se = make_field_struct(StructE)

    tensor_arg = qd.ndarray(qd.f32, shape=(4,))

    results = []

    # Benchmark 1: Many structs, all-field, no cache-busting arg (cache hits after warmup)
    results.append(run_benchmark(
        "many_structs_cached",
        lambda: kernel_many_structs(sa, sb, sc, sd, se),
    ))

    # Benchmark 2: Many structs + torch.Tensor (cache miss every call due to changing tensor)
    def step_structs_tensor():
        t = torch.zeros(4, dtype=torch.float32)
        kernel_structs_plus_tensor(sa, sb, sc, sd, se, t)

    results.append(run_benchmark("many_structs_cache_miss", step_structs_tensor))

    # Benchmark 3: Template annotations (baseline — no struct traversal at all)
    results.append(run_benchmark(
        "template_cached",
        lambda: kernel_template_annotations(sa, sb, sc, sd, se),
    ))

    # Benchmark 4: Template + tensor (baseline with cache-busting)
    def step_template_tensor():
        t = torch.zeros(4, dtype=torch.float32)
        kernel_template_plus_tensor(sa, sb, sc, sd, se, t)

    results.append(run_benchmark("template_cache_miss", step_template_tensor))

    return results


def main():
    parser = argparse.ArgumentParser(description="Quadrants kernel launch overhead benchmarks")
    parser.add_argument("--arch", default="cpu", choices=["cpu", "gpu"], help="Quadrants arch")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--wandb-project", default=None, help="W&B project to log results")
    parser.add_argument("--wandb-run-prefix", default="launch", help="W&B run name prefix")
    args = parser.parse_args()

    arch = qd.cpu if args.arch == "cpu" else qd.gpu
    qd.init(arch=arch)

    print(f"Running launch-overhead benchmarks on {args.arch}...")
    print(f"  N_WARMUP={N_WARMUP}, N_STEPS={N_STEPS}, N_TRIALS={N_TRIALS}")
    print()

    results = run_all()

    print("\n" + "=" * 70)
    print(f"{'Benchmark':<30} {'Median launches/s':>20}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<30} {r['median_launches_per_sec']:>20,.0f}")
    print("=" * 70)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")

    if args.wandb_project:
        import wandb

        revision = qd.__version__
        run_name = f"{args.wandb_run_prefix}-{revision}"
        run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={"arch": args.arch, "version": revision, "n_steps": N_STEPS, "n_trials": N_TRIALS},
            settings=wandb.Settings(x_disable_stats=True, console="off"),
        )
        for r in results:
            run.log({f"{r['name']}/launches_per_sec": r["median_launches_per_sec"]})
        # Also log as a summary table
        table = wandb.Table(columns=["benchmark", "launches_per_sec"])
        for r in results:
            table.add_data(r["name"], r["median_launches_per_sec"])
        run.log({"launch_overhead_results": table})
        run.finish()
        print(f"\nResults uploaded to W&B project '{args.wandb_project}'")


if __name__ == "__main__":
    main()
