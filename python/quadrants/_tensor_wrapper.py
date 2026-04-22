"""``Tensor`` wrapper — POC for ``hp/tensor-stork-17``.

A thin Python wrapper around an underlying ``Ndarray`` or ``ScalarField``
impl. The point is to make backend symmetry a *type* property rather than
something we police test by test:

- The wrapper exposes a fixed whitelisted surface; anything not on the
  whitelist is invisible to users (the upstream impl keeps its full
  surface for internal Quadrants use).
- Layout is owned by the wrapper, not the impl. The impl-side
  ``_qd_layout`` attribute survives only as the AST-rewrite trigger on
  ndarrays.
- Pickle, ``to_torch``, ``to_dlpack``, ``to_numpy`` all delegate to the
  impl, which is already symmetric across backends after stork-16.

For stork-17 this is **opt-in only**, exposed under a private name so we
don't disturb existing call sites. ``qd.tensor(...)`` still returns the
raw impl. The wrapper is exercised by the cache-fragmentation and host
indexing tests in stork-17 to de-risk the full migration in stork-18+.

See ``perso_hugh/doc/quadrants-tensor.md`` §8.11.
"""

from __future__ import annotations

import typing

# Late imports below intentionally deferred to method bodies to break
# the circular dependency through ``quadrants.lang``.
# pylint: disable=import-outside-toplevel


__all__ = [
    "Tensor",
]


def _is_identity(layout: typing.Optional[typing.Tuple[int, ...]]) -> bool:
    if layout is None:
        return True
    return tuple(layout) == tuple(range(len(layout)))


def _invert_perm(perm: typing.Tuple[int, ...]) -> typing.Tuple[int, ...]:
    """Inverse of a permutation: ``perm[invperm[i]] == i``."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


class Tensor:
    """Backend-agnostic tensor wrapper. POC, opt-in only.

    Holds a reference to an underlying impl (either an ``Ndarray`` or a
    ``ScalarField``) and forwards a whitelisted set of operations. The
    canonical layout permutation is owned by the wrapper; the impl-side
    ``_qd_layout`` tag is preserved only so the existing AST rewrite in
    ``ast_transformer.py`` keeps firing for ndarray-backed kernel code.

    Constructed explicitly via ``Tensor(impl)`` — there's no factory yet.
    The factory flip lives in stork-19.
    """

    __slots__ = ("_impl",)

    def __init__(self, impl: typing.Any) -> None:
        # Validate that ``impl`` is something we know how to wrap. Keep the
        # check string-based to avoid pulling lang imports into module load.
        from quadrants.lang._ndarray import Ndarray
        from quadrants.lang.field import Field

        if not isinstance(impl, (Ndarray, Field)):
            raise TypeError(
                f"Tensor(impl) requires an Ndarray or Field; got {type(impl).__name__}"
            )
        # Use object.__setattr__ to play nice with __slots__.
        object.__setattr__(self, "_impl", impl)

    # ------------------------------------------------------------------
    # Identity / debug
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        layout = self.layout
        layout_repr = "" if layout is None else f", layout={layout!r}"
        return (
            f"Tensor(shape={self.shape!r}, dtype={self.dtype!r}, "
            f"backend={self._backend_name()}{layout_repr})"
        )

    def _backend_name(self) -> str:
        from quadrants.lang._ndarray import Ndarray

        return "NDARRAY" if isinstance(self._impl, Ndarray) else "FIELD"

    # ------------------------------------------------------------------
    # Whitelisted introspection (no behavior change vs impl)
    # ------------------------------------------------------------------

    @property
    def shape(self) -> typing.Tuple[int, ...]:
        return tuple(self._impl.shape)

    @property
    def dtype(self) -> typing.Any:
        return self._impl.dtype

    @property
    def layout(self) -> typing.Optional[typing.Tuple[int, ...]]:
        # Forwards to the impl's ``layout`` property, which is symmetric
        # across backends after stork-16 (Ndarray reads ``_qd_layout``,
        # Field reads ``_qd_field_layout``).
        return self._impl.layout

    # ------------------------------------------------------------------
    # Internal escape hatch
    # ------------------------------------------------------------------

    def _unwrap(self) -> typing.Any:
        """Return the underlying impl. Used by the kernel-arg unwrap hook
        so ``@qd.kernel`` continues to see raw ``Ndarray`` / ``Field``
        types and the JIT cache key stays stable.
        """
        return self._impl
