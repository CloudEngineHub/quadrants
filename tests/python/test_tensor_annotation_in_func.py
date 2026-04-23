"""Tests for ``qd.Tensor`` as an annotation in ``@qd.func`` parameters
and struct fields.

Pre-stork-23, ``qd.Tensor`` only worked as an annotation for top-level
``@qd.kernel`` parameters (handled by ``_template_mapper_hotpath``).
Using it in a ``@qd.func`` parameter or as a struct field annotation
raised ``QuadrantsTypeError`` because ``_transform_func_arg`` had no
dispatch branch for the ``Tensor`` class.

Stork-23 adds that branch. These tests pin the invariant.
"""

import dataclasses

import numpy as np
import pytest

import quadrants as qd

from tests import test_utils

BACKENDS = [qd.Backend.FIELD, qd.Backend.NDARRAY]
BACKEND_IDS = ["field", "ndarray"]


# ---------------------------------------------------------------------------
# 1. qd.Tensor as a standalone @qd.func parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_annotated_func_param(backend):
    """A @qd.func with a qd.Tensor-annotated param, called from a kernel."""
    N = 8
    a = qd.tensor(qd.i32, shape=(N,), backend=backend)

    @qd.func
    def double_it(x: qd.Tensor, i: qd.i32):
        x[i] = x[i] * 2

    @qd.kernel
    def run(x: qd.Tensor):
        for i in range(N):
            x[i] = i + 1
            double_it(x, i)

    run(a)
    np.testing.assert_array_equal(a.to_numpy(), np.arange(1, N + 1) * 2)


# ---------------------------------------------------------------------------
# 2. @qd.data_oriented struct with qd.Tensor field, kernel template
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_struct_field_kernel_data_oriented(backend):
    """A struct with a qd.Tensor field passed to a kernel via template().
    FIELD uses @qd.data_oriented; NDARRAY uses @dataclasses.dataclass(frozen=True)."""
    N = 6
    t = qd.tensor(qd.i32, shape=(N,), backend=backend)

    if backend == qd.Backend.FIELD:

        @qd.data_oriented
        class S:
            def __init__(self, vals):
                self.vals = vals

        s = S(vals=t)
    else:

        @dataclasses.dataclass(frozen=True)
        class S:
            vals: qd.Tensor

        s = S(vals=t)

    @qd.kernel
    def fill(st: qd.template()):
        for i in range(N):
            st.vals[i] = i * 3

    fill(s)
    np.testing.assert_array_equal(t.to_numpy(), np.arange(N) * 3)


# ---------------------------------------------------------------------------
# 3. Struct with qd.Tensor field, func takes struct as template.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_struct_field_func_via_template(backend):
    """A @qd.func receives a struct (containing qd.Tensor fields) as
    a qd.template() arg."""
    N = 4

    @qd.data_oriented
    class S:
        def __init__(self, vals):
            self.vals = vals

    t = qd.tensor(qd.i32, shape=(N,), backend=backend)
    s = S(vals=t)

    @qd.func
    def inc_all(st: qd.template()):
        for i in range(N):
            st.vals[i] = st.vals[i] + 10

    @qd.kernel
    def run(st: qd.template()):
        for i in range(N):
            st.vals[i] = i
        inc_all(st)

    run(s)
    np.testing.assert_array_equal(t.to_numpy(), np.arange(N) + 10)


# ---------------------------------------------------------------------------
# 4. Tensor wrapper in struct field (unwrap path via build_Attribute).
#    qd.tensor() returns a Tensor wrapper (post stork-19), so this
#    exercises the Tensor-unwrap in build_Attribute.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_wrapper_in_struct_field_unwraps(backend):
    """When the struct field stores a qd.Tensor *wrapper*, the AST
    build_Attribute must unwrap it transparently."""
    N = 4

    @qd.data_oriented
    class S:
        def __init__(self, vals):
            self.vals = vals

    t = qd.tensor(qd.i32, shape=(N,), backend=backend)
    assert isinstance(t, qd.Tensor)
    s = S(vals=t)

    @qd.func
    def write(st: qd.template(), i: qd.i32, v: qd.i32):
        st.vals[i] = v

    @qd.kernel
    def run(st: qd.template()):
        for i in range(N):
            write(st, i, i * 7)

    run(s)
    np.testing.assert_array_equal(t.to_numpy(), np.arange(N) * 7)


# ---------------------------------------------------------------------------
# 5. Mixed qd.Tensor and scalar template fields in the same struct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_mixed_tensor_and_scalar_struct_fields(backend):
    """Struct with one qd.Tensor field and one scalar, both accessed
    in a @qd.func via template."""
    N = 4

    @qd.data_oriented
    class S:
        def __init__(self, tensor_field, scale):
            self.tensor_field = tensor_field
            self.scale = scale

    t = qd.tensor(qd.i32, shape=(N,), backend=backend)
    s = S(tensor_field=t, scale=5)

    @qd.func
    def scaled_fill(st: qd.template(), i: qd.i32):
        st.tensor_field[i] = i * st.scale

    @qd.kernel
    def run(st: qd.template()):
        for i in range(N):
            scaled_fill(st, i)

    run(s)
    np.testing.assert_array_equal(t.to_numpy(), np.arange(N) * 5)


# ---------------------------------------------------------------------------
# 7. qd.Tensor func param with 2D tensor and layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKEND_IDS)
@test_utils.test(arch=qd.cpu)
def test_tensor_func_param_2d_with_layout(backend):
    """qd.Tensor func param with a 2D layout-tagged tensor."""
    M, N = 3, 4
    a = qd.tensor(qd.i32, shape=(M, N), backend=backend, layout=(1, 0))

    @qd.func
    def fill_row(x: qd.Tensor, row: qd.i32):
        for j in range(N):
            x[row, j] = row * 100 + j

    @qd.kernel
    def run(x: qd.Tensor):
        for i in range(M):
            fill_row(x, i)

    run(a)
    arr = a.to_numpy()
    assert arr.shape == (M, N)
    for i in range(M):
        for j in range(N):
            assert arr[i, j] == i * 100 + j, f"mismatch at [{i},{j}]"
