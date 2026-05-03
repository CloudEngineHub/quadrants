"""Standalone Genesis rigid-step repro for the dynamic-loop adstack perf plan.

Builds the same scene shape as `tests/test_grad.py::test_differentiable_rigid[cpu]` (single rigid box,
no contact, no constraint, gravity-only) and runs one outer iteration of forward + backward to surface
the hot reverse-grad kernels under a profiler / IR dump.

Run:
    QD_OFFLINE_CACHE=0 GS_ENABLE_NDARRAY=0 python repro_genesis_rigid_step.py

With diagnostics:
    QD_OFFLINE_CACHE=0 GS_ENABLE_NDARRAY=0 QD_ELIM_DIAG=1 python repro_genesis_rigid_step.py
"""

import time

import genesis as gs
import torch


def main():
    gs.init(backend=gs.cpu, debug=False, logger_verbose_time=False)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=1,
            requires_grad=True,
            gravity=(0, 0, -1),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False,
            enable_self_collision=False,
            enable_joint_limit=False,
            disable_constraint=True,
            use_contact_island=False,
            use_hibernation=False,
        ),
        show_viewer=False,
    )
    box = scene.add_entity(gs.morphs.Box(pos=(0, 0, 0), size=(0.1, 0.1, 0.2)))
    scene.build()

    horizon = 100
    init_pos = gs.tensor([0.3, 0.1, 0.28], requires_grad=True)
    init_quat = gs.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)

    n_warmup = 1
    n_bench = 3

    for trial in range(n_warmup + n_bench):
        scene.reset()
        box.set_pos(init_pos)
        box.set_quat(init_quat)
        t0 = time.perf_counter()
        for _ in range(horizon):
            scene.step()
        box_state = box.get_state()
        loss = torch.abs(box_state.pos).sum() + torch.abs(box_state.quat).sum()
        loss.backward()
        elapsed = time.perf_counter() - t0
        if trial >= n_warmup:
            fps = horizon / elapsed
            print(
                f"trial {trial - n_warmup}: {horizon} substeps + 1 backward in {elapsed * 1000:.1f}ms ({fps:.1f} FPS)"
            )


if __name__ == "__main__":
    main()
