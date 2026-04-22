"""Pin host-side ``t[i, j]`` semantics on layout-tagged tensors.

Background
----------
The canonical -> physical index permutation for layout-tagged tensors is
implemented in ``ast_transformer.py`` (``build_Subscript`` and
``build_struct_for``), which only fires inside ``@qd.kernel`` bodies.
Python-scope (host) ``__getitem__`` / ``__setitem__`` on the *bare*
``Ndarray`` is **not** layout-aware — the index hits the host accessor
directly. Field is layout-aware for free because its host accessor walks
the SNode tree (which applies ``order=``).

Per the user-stated rule that ``layout=`` must be invisible to users
("anything that doesn't work with non-identity layout is also useless"),
host-side indexing has to return the *canonical* element, identical to
``a.to_numpy()[i, j]``.

The ``qd._Tensor`` wrapper (stork-18) owns this fix: on a layout-tagged
ndarray it translates the canonical user key to physical coords before
hitting the impl accessor; on a field it simply delegates. Both paths
give the user the canonical view, symmetric across backends.

This file covers two contracts:

- *bare-impl* (legacy surface). Field passes for free; Ndarray
  non-identity-layout is the pre-existing gotcha B bug, marked ``xfail``
  until ``qd.tensor()`` is flipped to return wrappers in stork-19.
- *wrapped* (``qd._Tensor(impl)``). Must pass on both backends at every
  layout (wrapped is the canonical-view contract the user actually sees
  once we flip the default).
"""

import itertools

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]

_LAYOUTS_RANK2 = [(0, 1), (1, 0)]
_LAYOUTS_RANK3 = [(0, 1, 2), (2, 1, 0), (2, 0, 1), (1, 2, 0)]


def _is_identity(layout):
    return layout == tuple(range(len(layout)))


def _xfail_if_ndarray_non_identity(backend, layout):
    """Apply ``pytest.xfail`` for the (Ndarray + non-identity layout) cell.

    Confirmed broken on ``hp/tensor-stork-17`` (this is the gotcha B
    reproduction). Field passes for free because its host accessor walks
    the SNode tree, which already applies the ``order=`` permutation.
    The fix lands in the upcoming ``Tensor`` wrapper (§8.11) which will
    permute host indices in ``__getitem__`` / ``__setitem__``.
    """
    if backend is qd.Backend.NDARRAY and not _is_identity(layout):
        pytest.xfail(
            "gotcha B: Ndarray host-side __getitem__/__setitem__ is not "
            "layout-aware; canonical->physical permutation only fires "
            "inside @qd.kernel via ast_transformer.py. Fixed in the "
            "Tensor wrapper (see design doc §8.11)."
        )


def _fill_via_from_numpy(a, canonical_shape):
    """Populate ``a`` with distinct, position-encoding values via the
    layout-aware ``from_numpy`` path. Returns the source numpy array so
    the test can use it as the canonical reference."""
    src = np.arange(int(np.prod(canonical_shape)), dtype=np.int32).reshape(canonical_shape)
    a.from_numpy(src)
    return src


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_host_getitem_canonical_rank2(backend, layout):
    """``a[i, j]`` at host scope must equal ``a.to_numpy()[i, j]``."""
    _xfail_if_ndarray_non_identity(backend, layout)
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    src = _fill_via_from_numpy(a, canonical)

    np_view = a.to_numpy()
    np.testing.assert_array_equal(np_view, src)  # to_numpy is canonical (already pinned)

    for ci in itertools.product(*(range(d) for d in canonical)):
        host = a[ci]
        canon = int(src[ci])
        assert int(host) == canon, (
            f"host indexing leaked physical layout: a{list(ci)} = {int(host)}, "
            f"canonical (to_numpy/source) = {canon}, layout={layout}"
        )


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK3)
@test_utils.test(arch=qd.cpu)
def test_host_getitem_canonical_rank3(backend, layout):
    _xfail_if_ndarray_non_identity(backend, layout)
    canonical = (2, 3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    src = _fill_via_from_numpy(a, canonical)

    np_view = a.to_numpy()
    np.testing.assert_array_equal(np_view, src)

    for ci in itertools.product(*(range(d) for d in canonical)):
        host = a[ci]
        canon = int(src[ci])
        assert int(host) == canon, (
            f"host indexing leaked physical layout: a{list(ci)} = {int(host)}, "
            f"canonical (to_numpy/source) = {canon}, layout={layout}"
        )


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_host_setitem_canonical_rank2(backend, layout):
    """``a[i, j] = v`` at host scope must place ``v`` at canonical
    coordinate ``(i, j)``, i.e. ``a.to_numpy()[i, j] == v``.
    """
    _xfail_if_ndarray_non_identity(backend, layout)
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)

    expected = np.zeros(canonical, dtype=np.int32)
    for n, ci in enumerate(itertools.product(*(range(d) for d in canonical))):
        v = 1000 + n
        a[ci] = v
        expected[ci] = v

    np.testing.assert_array_equal(a.to_numpy(), expected)


# ----------------------------------------------------------------------
# Wrapped-tensor tests: the canonical-view contract must hold at *every*
# (backend, layout) cell when the user interacts via ``qd._Tensor``.
# These are the tests that exercise the stork-18 host-indexing fix.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapped_host_getitem_canonical_rank2(backend, layout):
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    src = _fill_via_from_numpy(a, canonical)

    t = qd._Tensor(a)
    assert t.shape == canonical
    assert t.layout == (None if _is_identity(layout) else layout)

    for ci in itertools.product(*(range(d) for d in canonical)):
        host = t[ci]
        canon = int(src[ci])
        assert int(host) == canon, (
            f"wrapped host indexing broke canonical view: t{list(ci)} = "
            f"{int(host)}, canonical = {canon}, backend={backend!r}, "
            f"layout={layout}"
        )


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK3)
@test_utils.test(arch=qd.cpu)
def test_wrapped_host_getitem_canonical_rank3(backend, layout):
    canonical = (2, 3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    src = _fill_via_from_numpy(a, canonical)

    t = qd._Tensor(a)
    for ci in itertools.product(*(range(d) for d in canonical)):
        host = t[ci]
        canon = int(src[ci])
        assert int(host) == canon, (
            f"wrapped host indexing broke canonical view: t{list(ci)} = "
            f"{int(host)}, canonical = {canon}, backend={backend!r}, "
            f"layout={layout}"
        )


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", _LAYOUTS_RANK2)
@test_utils.test(arch=qd.cpu)
def test_wrapped_host_setitem_canonical_rank2(backend, layout):
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)
    t = qd._Tensor(a)

    expected = np.zeros(canonical, dtype=np.int32)
    for n, ci in enumerate(itertools.product(*(range(d) for d in canonical))):
        v = 2000 + n
        t[ci] = v
        expected[ci] = v

    np.testing.assert_array_equal(a.to_numpy(), expected)
    # And reading back through the wrapper matches too.
    for ci in itertools.product(*(range(d) for d in canonical)):
        assert int(t[ci]) == int(expected[ci])
