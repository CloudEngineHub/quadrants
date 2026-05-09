# type: ignore

import platform
import warnings

from quadrants._lib import core as _qd_core
from quadrants.lang import impl


def arch_uses_spv(arch):
    return arch == _qd_core.vulkan or arch == _qd_core.metal


def _is_metal_like(arch):
    if arch == _qd_core.metal:
        return True
    if arch == _qd_core.vulkan and platform.system() == "Darwin":
        return True
    return False


def mem_fence():
    arch = impl.get_runtime().prog.config().arch
    if arch == _qd_core.cuda or arch == _qd_core.amdgpu:
        return impl.call_internal("grid_mem_fence", with_runtime_context=False)
    if arch_uses_spv(arch):
        return impl.call_internal("gridMemoryBarrier", with_runtime_context=False)
    raise ValueError(f"qd.simt.grid.mem_fence is not supported for arch {arch}")


def publish(target, value):
    """Device-scope publish: store *value* into *target* with cross-workgroup
    visibility guarantees.  All prior plain stores by the calling thread are
    ordered before this store and become visible to any thread that
    :func:`observe`\\s *target* (or any later address).

    On CUDA, AMDGPU and native Vulkan the implementation is a device-scope
    memory fence followed by a plain store.  On Metal (and Vulkan-on-macOS,
    which is MoltenVK lowering to MSL) the store goes through
    ``atomic_exchange`` so that cross-workgroup ordering is backed by the
    MSL atomic-ordering spec in addition to the empirical fence behaviour.

    Args:
        target: A writable field subscript (e.g. ``flags[i]``).
        value:  The value to publish.  Must be promotable to *target*'s dtype.
    """
    from quadrants.lang import ops

    mem_fence()
    arch = impl.get_runtime().prog.config().arch
    if _is_metal_like(arch):
        ops.atomic_exchange(target, value)
    else:
        ops.assign(target, value)


def observe(target):
    """Device-scope observe: atomically read *target* and ensure that
    subsequent plain loads by the calling thread see all stores that were
    ordered before the corresponding :func:`publish`.

    The read is implemented as ``atomic_or(target, 0)`` (a no-op RMW that
    returns the current value) on every backend, which both defeats LICM in
    spin-wait loops and satisfies Metal's requirement that cross-workgroup
    ordering go through atomic operations.  A device-scope memory fence
    follows the atomic read.

    Args:
        target: A readable field subscript (e.g. ``flags[i]``).
                Must be an integer dtype (``atomic_or`` is not defined
                on floats).

    Returns:
        The current value of *target*.
    """
    from quadrants.lang import ops

    val = ops.atomic_or(target, 0)
    mem_fence()
    return val


def memfence():
    warnings.warn(
        "qd.simt.grid.memfence() is deprecated; use qd.simt.grid.mem_fence() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return mem_fence()
