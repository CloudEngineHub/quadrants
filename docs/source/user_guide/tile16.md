# Tile16x16: register-resident 16x16 tiles

`Tile16x16` provides a 16x16 matrix tile that lives entirely in registers, distributed across 16 threads in a subgroup (warp). Each thread holds one row as 16 scalar registers. Cross-thread communication uses subgroup shuffles — no shared memory needed.

This is useful for implementing blocked linear algebra kernels (Cholesky, triangular solve, etc.) where you want to keep working data in registers for maximum throughput.

Tile16x16 runs on all GPU backends supported by Quadrants: CUDA, AMD, Metal, and Vulkan. It builds on `qd.simt.subgroup.shuffle`, which is cross-platform — no vendor-specific libraries required.

## Quick start

```python
import quadrants as qd
from quadrants.lang.simt._tile16 import _make_tile16x16

Tile = _make_tile16x16(qd.f32)

@qd.func
def my_blocked_op(A, row0, col0, eps):
    t = Tile.zeros()
    t[:] = A[row0:row0+16, col0:col0+16]
    t.cholesky_(eps)
    A[row0:row0+16, col0:col0+16] = t
```

## Creating a tile

```python
Tile = _make_tile16x16(qd.f32)   # or qd.f64

t = Tile.zeros()       # all zeros
t = Tile.eye()         # 16x16 identity
```

`_make_tile16x16` returns a `qd.dataclass` type whose 16 fields (`r0`–`r15`) are the scalar registers for one row. The result is cached per dtype.

## Loading and storing

Load/store transfers data between a tile and device memory arrays using slice syntax. Each thread accesses row `row0 + tid`, where `tid` is the thread's subgroup lane index (obtained internally via `subgroup.invocation_id()`).

### 2D arrays

```python
t = Tile.zeros()
t[:] = arr[row0:row0+16, col0:col0+16]    # load
arr[row0:row0+16, col0:col0+16] = t       # store
```

### 3D arrays

For arrays with a leading batch dimension (e.g. `H[batch, row, col]`):

```python
t = Tile.zeros()
t[:] = arr[batch, row0:row0+16, col0:col0+16]    # load
arr[batch, row0:row0+16, col0:col0+16] = t       # store
```

The load/store automatically clamps to the array's shape, so out-of-bounds columns are left as zero (load) or skipped (store).

The `[:]` on the load LHS is required — it distinguishes an in-place tile load from a variable rebinding. The store side does not need `[:]` because the array subscript on the LHS already triggers the correct assignment path.

## Identity initialization

For padding partial tiles in blocked algorithms:

```python
t = Tile.eye()    # create a new identity tile
t._eye_()         # or reset an existing tile to identity in-place
```

Each thread sets its diagonal element to 1.0 and all others to 0.0.

## Rank-1 updates

```python
t -= qd.outer(v, v)    # t -= v @ v^T  (symmetric)
t -= qd.outer(a, b)    # t -= a @ b^T  (general)
```

Each thread provides its element(s) of the vector(s). The outer product is computed via subgroup shuffles and subtracted from the tile in-place. `qd.outer(a, b)` returns a deferred proxy — it is only valid as the RHS of `-=` on a Tile16x16. Composition like `qd.outer(a, b) + qd.outer(c, d)` raises `TypeError`.

### Loading column vectors for outer products

Column vectors can be loaded from arrays using slice syntax:

```python
t -= qd.outer(arr[row0:row0+16, col], arr[row0:row0+16, col])
```

Each thread loads one element from the column. Out-of-range threads get zero. This also works with 3D arrays:

```python
t -= qd.outer(arr[batch, row0:row0+16, col], arr[batch, row0:row0+16, col])
```

## Cholesky factorization

```python
t.cholesky_(eps)
```

Factorizes the tile in-place: replaces the lower triangle with `L` such that `L @ L^T ≈ A`. The `eps` parameter clamps the diagonal to avoid numerical issues with near-singular matrices. After this call, the lower triangle of `t` contains `L`.

## Triangular solve

```python
L.solve_triangular_(B)
```

Solves `X @ L^T = B` in-place, replacing `B` with `X`. `L` must be a lower-triangular tile (e.g. from `cholesky_()`). Only `lower=True` is supported; passing `lower=False` raises `TypeError`.

## Kernel structure

Tile16x16 requires exactly 16 threads per subgroup. Use `qd.loop_config(block_dim=16)` before the parallel loop:

```python
@qd.kernel
def my_kernel(A: qd.types.NDArray[qd.f32, 3]):
    qd.loop_config(block_dim=16)
    for i in range(A.shape[0]):
        t = Tile.zeros()
        t[:] = A[i, 0:16, 0:16]
        t.cholesky_(1e-6)
        A[i, 0:16, 0:16] = t
```

## f64 support

Pass `qd.f64` to the factory for double precision:

```python
Tile64 = _make_tile16x16(qd.f64)
```

Not all GPU backends support f64. Use `test_utils.skip_if_f64_unsupported()` in tests.

## Method reference

| Operation | Description |
|-----------|-------------|
| `Tile.zeros()` | Create a zero-initialized tile |
| `Tile.eye()` | Create an identity tile |
| `Tile.SIZE` | Tile dimension constant (16) |
| `t[:] = arr[r0:r1, c0:c1]` | Load from 2D array |
| `t[:] = arr[i, r0:r1, c0:c1]` | Load from 3D array |
| `arr[r0:r1, c0:c1] = t` | Store to 2D array |
| `arr[i, r0:r1, c0:c1] = t` | Store to 3D array |
| `t._eye_()` | Set to identity matrix (in-place) |
| `t -= qd.outer(v, v)` | Symmetric rank-1 subtract |
| `t -= qd.outer(a, b)` | General rank-1 subtract |
| `t.cholesky_(eps)` | In-place Cholesky factorization |
| `L.solve_triangular_(B)` | Triangular solve (in-place on B) |
