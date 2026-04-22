"""Pin host-side ``t[i, j]`` semantics on layout-tagged tensors.

Background
----------
The canonical -> physical index permutation for layout-tagged tensors is
implemented in ``ast_transformer.py`` (``build_Subscript`` and
``build_struct_for``), which only fires inside ``@qd.kernel`` bodies.
Python-scope (host) ``__getitem__`` / ``__setitem__`` on the underlying
``Ndarray`` / ``ScalarField`` is *not* layout-aware: the index is passed
straight to the host accessor (ndarray) or to the snode-tree (field).

Per the user-stated rule that ``layout=`` must be invisible to users
("anything that doesn't work with non-identity layout is also useless"),
host-side indexing has to return the *canonical* element, identical to
``a.to_numpy()[i, j]``.

These tests verify that contract today. Failures here are the reproduction
of "gotcha B" from the design doc (§8.11) and motivate the host-side
permutation in the upcoming ``Tensor`` wrapper.

Each test compares ``a[ci]`` to ``a.to_numpy()[ci]`` at *every* canonical
coordinate (small shapes, exhaustive). Both must agree; otherwise host
indexing leaks the physical layout to the user.
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
    canonical = (3, 4)
    a = qd.tensor(qd.i32, shape=canonical, backend=backend, layout=layout)

    expected = np.zeros(canonical, dtype=np.int32)
    for n, ci in enumerate(itertools.product(*(range(d) for d in canonical))):
        v = 1000 + n
        a[ci] = v
        expected[ci] = v

    np.testing.assert_array_equal(a.to_numpy(), expected)
