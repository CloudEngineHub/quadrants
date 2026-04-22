"""Skeleton tests for the opt-in ``qd._Tensor`` wrapper (stork-17 POC).

These tests do **not** exercise kernel-arg unwrap or layout-aware host
indexing yet; that's the next step. They only pin the wrapper's
introspection surface so we can wire up the kernel hook in a follow-up
commit on the same branch without re-reasoning the basics.
"""

import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_wrapper_construction_forwards_basic_attrs(backend):
    impl = qd.tensor(qd.f32, shape=(3, 4), backend=backend)
    t = qd._Tensor(impl)
    assert t.shape == (3, 4)
    assert t.dtype == qd.f32
    assert t.layout is None
    assert t._unwrap() is impl


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@pytest.mark.parametrize("layout", [(0, 1), (1, 0)])
@test_utils.test(arch=qd.cpu)
def test_wrapper_layout_is_canonical_and_introspectable(backend, layout):
    impl = qd.tensor(qd.f32, shape=(3, 4), backend=backend, layout=layout)
    t = qd._Tensor(impl)
    # ``shape`` is always canonical; ``layout`` reflects the user-supplied
    # permutation (or ``None`` for identity, normalised at the impl layer).
    assert t.shape == (3, 4)
    if layout == (0, 1):
        assert t.layout is None
    else:
        assert t.layout == layout


def test_wrapper_rejects_non_tensor():
    with pytest.raises(TypeError, match="Tensor.*requires"):
        qd._Tensor(42)
    with pytest.raises(TypeError, match="Tensor.*requires"):
        qd._Tensor("not a tensor")


@test_utils.test(arch=qd.cpu)
def test_wrapper_repr_includes_backend_and_layout():
    impl = qd.tensor(qd.i32, shape=(2, 3), backend=qd.Backend.NDARRAY, layout=(1, 0))
    r = repr(qd._Tensor(impl))
    assert "NDARRAY" in r
    assert "(2, 3)" in r
    assert "(1, 0)" in r
