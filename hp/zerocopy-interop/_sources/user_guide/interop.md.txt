# Numpy and Torch interop

Quadrants provides interop with both numpy and PyTorch. There are three mechanisms:
- **Copy-based**: convert data between quadrants fields/ndarrays and numpy arrays or torch tensors
- **Zero-copy via DLPack**: obtain a torch tensor or numpy array that aliases the underlying Quadrants memory
- **Direct pass-through**: pass torch tensors directly into kernels as ndarray arguments (zero-copy)

## Copy-based interop

### Fields

Fields support `to_numpy()`, `from_numpy()`, `to_torch()`, and `from_torch()`:

```python
import numpy as np
import quadrants as qd

qd.init(arch=qd.gpu)

f = qd.field(qd.f32, shape=(4, 4))

# numpy -> field
arr = np.ones((4, 4), dtype=np.float32) * 3.0
f.from_numpy(arr)

# field -> numpy
result = f.to_numpy()
print(result[0, 0])  # 3.0
```

With torch:

```python
import torch

f = qd.field(qd.f32, shape=(4, 4))

# torch -> field
t = torch.ones(4, 4, dtype=torch.float32) * 5.0
f.from_torch(t)

# field -> torch
result = f.to_torch(device="cpu")
print(result[0, 0])  # 5.0
```

### Ndarrays

Ndarrays support `to_numpy()` and `from_numpy()`:

```python
a = qd.ndarray(qd.i32, shape=(10,))

# numpy -> ndarray
arr = np.arange(10, dtype=np.int32)
a.from_numpy(arr)

# ndarray -> numpy
result = a.to_numpy()
```

### Shape requirements

The shape of the numpy array or torch tensor must match the shape of the field or ndarray exactly. For matrix and vector fields, the element dimensions are appended to the shape:

```python
# A field of 3x2 matrices with shape (4,) has numpy shape (4, 3, 2)
m = qd.Matrix.field(3, 2, qd.f32, shape=(4,))
arr = np.zeros((4, 3, 2), dtype=np.float32)
m.from_numpy(arr)
```

## Zero-copy interop via DLPack

Quadrants' zero-copy interop has been designed with **PyTorch as the first-class user interface**: support, defaults, and supported-dtype/backend matrices are driven by what PyTorch can consume cleanly via DLPack. NumPy is supported on CPU backends as a free side benefit of the same DLPack capsule. Several of the limitations below (e.g. the Apple Metal `torch >= 2.9.2` requirement, or the 0-dim `ScalarField` carve-out) are inherited from PyTorch's current DLPack importer rather than from Quadrants itself.

`to_torch()` and `to_numpy()` accept a keyword-only `copy` argument that controls whether the returned tensor/array is an independent copy of the data or a zero-copy view that aliases the underlying Quadrants memory.

```python
f = qd.field(qd.f32, shape=(1024,))

view  = f.to_torch(copy=False)  # zero-copy view: aliases f's memory
clone = f.to_torch(copy=True)   # independent copy
auto  = f.to_torch()            # default: copy=None (see below)
```

Modifications via the view are visible to subsequent Quadrants kernel reads, and vice-versa. The view stays valid until the underlying storage is reallocated -- typically on `qd.init()` or `qd.reset()`, after which a fresh call to `to_torch()` / `to_numpy()` returns a new view.

### When zero-copy is available

Zero-copy uses [DLPack](https://github.com/dmlc/dlpack) and requires:

- a backend with DLPack support: `cpu` (`x64`/`arm64`), `cuda`, `amdgpu`, or `metal`. Vulkan is not supported: Vulkan-backed DLPack tensors are not processed by many well-known scientific computing libraries (torch, numpy, etc.);
- a DLPack-supported dtype: `i32`, `i64`, `f32`, `f64`, `u1` (other dtypes such as `f16`, `u8`, `u16` fall back to the kernel-copy path);
- on Apple Metal, `torch >= 2.9.2` for fields (required for DLPack `bytes_offset` on MPS; see [pytorch/pytorch#168193](https://github.com/pytorch/pytorch/pull/168193));
- 0-dim `ScalarField` instances are not zero-copyable on any backend (PyTorch DLPack `bytes_offset` limitation);
- members of an AOS `StructField` (the default `Struct.field(..., layout=Layout.AOS)`) are not zero-copyable yet (see [Struct fields](#struct-fields) below); members of an SOA `StructField` (`layout=Layout.SOA`) **are** zero-copyable individually.

Zero-copy `to_numpy()` additionally requires a CPU backend, because numpy arrays cannot reference GPU memory.

### Semantics of `copy`

| Value | Behaviour |
|---|---|
| `None` (default) | Zero-copy view if supported, otherwise an independent copy. Never raises for support reasons. |
| `False` | Zero-copy view, or `ValueError` if zero-copy is unsupported for this backend/dtype. |
| `True` | Always returns an independent copy (clones the zero-copy view if available; otherwise allocates and runs the kernel-copy path). |

`copy=True` always returns a buffer that is safe to mutate without affecting the field/ndarray.

### Examples

```python
import quadrants as qd

qd.init(arch=qd.cuda)

f = qd.field(qd.f32, shape=(1024,))
f.fill(1.0)

view = f.to_torch(copy=False)
view *= 2.0           # mutates f's underlying memory directly
qd.sync()             # not strictly required; safe pattern

print(f[0])           # 2.0
```

Round-trip with NumPy on a CPU backend:

```python
qd.init(arch=qd.cpu)

a = qd.ndarray(qd.i32, shape=(8,))
a.from_numpy(np.arange(8, dtype=np.int32))

view = a.to_numpy(copy=False)
view[0] = 100
print(a.to_numpy()[0])   # 100  (default copy=None still returns the same view here)
```

### Caching

Zero-copy views are cached on the source object: repeated calls to `to_torch()` / `to_numpy()` on the same field/ndarray return the same underlying tensor without rebuilding the DLPack capsule. The cache is invalidated automatically when the runtime is reinitialised or reset.

```python
v1 = f.to_torch(copy=False)
v2 = f.to_torch(copy=False)
assert v1.data_ptr() == v2.data_ptr()   # same view

qd.reset()
qd.init(arch=qd.cuda)
f2 = qd.field(qd.f32, shape=(1024,))
v3 = f2.to_torch(copy=False)            # fresh view; v1/v2 must not be used
```

### Axis layout with `layout=`

`to_torch()` and `to_numpy()` accept a keyword `layout: tuple[int, ...] | None = None` that selects the axis order of the returned view. When set, `layout` is interpreted as an axis permutation in the same convention as `torch.movedim(src, dst)` / `numpy.moveaxis(src, dst)`: the i-th element of the tuple says which input axis ends up at output position `i`. The default `layout=None` returns the natural view (Quadrants' native order, batch axis last).

The permuted view is materialised once via a single `movedim` (or `moveaxis`) from the natural-layout view and **cached per perm-tuple key** alongside the natural view, so repeated calls in a hot loop with the same `layout=` reduce to a tuple-equality check + cached-tensor return.

```python
f = qd.field(qd.f32, shape=(7, 4096))   # 7 components, 4096 envs

natural     = f.to_torch(copy=False)                            # shape (7, 4096)
batch_first = f.to_torch(copy=False, layout=(1, 0))             # shape (4096, 7)

assert natural.data_ptr() == batch_first.data_ptr()             # same underlying memory
```

For an n-d field the "move last axis to front" perm common in batched simulations and RL is `(n - 1, *range(n - 1))`:

```python
f3d = qd.field(qd.f32, shape=(3, 7, 4096))
n = len(f3d.shape)
batch_first = f3d.to_torch(copy=False, layout=(n - 1, *range(n - 1)))   # (4096, 3, 7)
```

Caching matters on CPU, where the Python-side `to_torch().movedim(...)` pattern is dispatch-heavy in a per-step loop:

```python
# Hot loop -- prefer this:
PERM = (n - 1, *range(n - 1))
for _ in range(n_steps):
    dst = f.to_torch(copy=False, layout=PERM)        # cached, one tuple-eq + return on the hit path
    ...

# over the equivalent but slower:
for _ in range(n_steps):
    dst = f.to_torch(copy=False).movedim(-1, 0)      # fresh torch dispatch every step
    ...
```

`layout=` composes with `copy=True` (clones the cached layout view), `device=` (transfers the cached layout view), `keep_dims` (matrix fields only), and the lifetime / synchronisation rules described above.

For 0-D and 1-D fields, every layout collapses to the same tensor (any 1-element perm is a no-op; for 1-D fields the only legal perm is `(0,)`).

Cache structure: each field holds a `dict[tuple[int, ...], Tensor]` of permuted views plus a single-slot cache for the most recently requested key. In typical hot loops a callsite always passes the same perm tuple, so the slot hits ~100 % of the time. The dict entries (and the slot) are cleared together with the natural-layout view on `qd.reset()` / `qd.init()`.

### Apple Metal: synchronisation

On Apple Metal, Quadrants and PyTorch MPS use separate Metal command queues. Quadrants kernel writes are made visible to the MPS-backed view via an automatic `qd.sync()` on every `to_torch()` / `to_numpy()` call. When you ask Quadrants for an independent buffer with `copy=True`, Quadrants additionally calls `torch.mps.synchronize()` after cloning so the returned tensor reflects the latest device writes:

```python
qd.init(arch=qd.metal)
f = qd.field(qd.f32, shape=(64,))

run_kernel(f)                       # queues writes on the Quadrants Metal stream
view = f.to_torch()                 # qd.sync() runs internally; view sees the writes
copy = f.to_torch(copy=True)        # qd.sync() + torch.mps.synchronize() run internally
```

`view.clone()`, called by you on a tensor you already hold, is a plain PyTorch op and does **not** go through Quadrants -- it neither calls `qd.sync()` nor `torch.mps.synchronize()`. If you need that, either go through `f.to_torch(copy=True)` (which does both internally) or call `torch.mps.synchronize()` yourself before / after the clone.

The reverse direction (PyTorch writes to a zero-copy view, then a Quadrants kernel reads from the same field) is **not** automatically synchronised. Because Quadrants and PyTorch MPS submit work to separate Metal command queues, a kernel launched immediately after a torch write may execute before the torch write has actually committed to memory:

```python
qd.init(arch=qd.metal)
f = qd.field(qd.f32, shape=(64,))

view = f.to_torch(copy=False)
view.zero_()                     # queued on the torch MPS stream
my_kernel(f)                     # may run BEFORE view.zero_() commits!

torch.mps.synchronize()          # required to flush the torch MPS stream first
my_kernel(f)                     # now safe
```

This is intentional: forcing a sync on every Quadrants kernel that touches a previously-zerocopied field would be very expensive in workloads that batch many torch ops and many kernels back-to-back. If you mutate fields from torch and then read them from a Quadrants kernel on Metal, call `torch.mps.synchronize()` once between the torch ops and the kernels.

### Lifetime caveats

A zero-copy view becomes invalid when the underlying Quadrants storage is freed. This happens on `qd.reset()` and `qd.init()`. Holding a `copy=False` tensor across either is undefined behaviour:

```python
view = f.to_torch(copy=False)
qd.reset()
view[0]                 # undefined: view aliases freed memory
```

If you need a tensor that outlives the runtime, use `copy=True` (or the default `None`, which produces a view but is bound to the same caveat once you opt into zero-copy).

### Struct fields

`StructField.to_torch()` and `StructField.to_numpy()` return a dictionary mapping each member name to a tensor / array; the `copy` argument is propagated to each member, so zero-copy availability is decided per member. The relevant axis is the SNode layout chosen at construction:

- **AOS** (default `Struct.field(..., layout=Layout.AOS)`): all members share the struct cell, e.g. `Struct.field({"a": i32, "b": f32}, shape=(N,))` stores `[a0, b0, a1, b1, ...]` in memory, with stride `sizeof(cell)` between consecutive `a`'s. Quadrants' C++ DLPack export does not currently emit cell-stride-aware views for individual members (it computes contiguous strides at the member dtype size, which would interleave neighbouring members' bytes), so AOS members fall back to a kernel copy and `copy=False` raises on each AOS member.
- **SOA** (`Struct.field(..., layout=Layout.SOA)`): each member sits in its own dense SNode subtree with contiguous storage, so members are zero-copyable individually under the usual backend / dtype rules. `copy=False` succeeds and returns aliasing views.

```python
S_aos = qd.Struct.field({"pos": qd.f32, "vel": qd.f32}, shape=(16,))   # AOS (default)
d_aos = S_aos.to_torch()                                                # dict of kernel copies
d_aos["pos"][0] = 1.0                                                   # does NOT write back

S_soa = qd.Struct.field({"pos": qd.f32, "vel": qd.f32}, shape=(16,),
                        layout=qd.Layout.SOA)
d_soa = S_soa.to_torch(copy=False)                                      # dict of zero-copy views
d_soa["pos"][0] = 1.0                                                   # writes through to S_soa.pos
```

## Direct torch tensor pass-through

Torch tensors can be passed directly into kernels where `qd.types.ndarray()` parameters are expected. The kernel reads from and writes to the torch tensor directly:

```python
import torch
import quadrants as qd

qd.init(arch=qd.gpu)

@qd.kernel
def square(inp: qd.types.ndarray(), out: qd.types.ndarray()) -> None:
    for i in range(32):
        out[i] = inp[i] * inp[i]

x = torch.ones(32, dtype=torch.float32) * 3.0
y = torch.zeros(32, dtype=torch.float32)
square(x, y)
print(y[0])  # 9.0
```

This also works with CUDA tensors when running on a CUDA backend:

```python
x = torch.ones(32, dtype=torch.float32, device="cuda:0") * 3.0
y = torch.zeros(32, dtype=torch.float32, device="cuda:0")
square(x, y)
```

### Integration with torch.autograd

Since torch tensors can be passed directly into kernels, you can integrate Quadrants kernels into PyTorch's autograd system by wrapping them in a `torch.autograd.Function`:

```python
@qd.kernel
def forward_kernel(t: qd.types.ndarray(), o: qd.types.ndarray()) -> None:
    for i in range(32):
        o[i] = t[i] * t[i]

@qd.kernel
def backward_kernel(t_grad: qd.types.ndarray(), t: qd.types.ndarray(), o_grad: qd.types.ndarray()) -> None:
    for i in range(32):
        t_grad[i] = 2 * t[i] * o_grad[i]

class Sqr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        outp = torch.zeros_like(inp)
        ctx.save_for_backward(inp)
        forward_kernel(inp, outp)
        return outp

    @staticmethod
    def backward(ctx, outp_grad):
        outp_grad = outp_grad.contiguous()
        inp_grad = torch.zeros_like(outp_grad)
        (inp,) = ctx.saved_tensors
        backward_kernel(inp_grad, inp, outp_grad)
        return inp_grad

x = torch.tensor([2.0] * 32, requires_grad=True)
loss = Sqr.apply(x).sum()
loss.backward()
print(x.grad[0])  # 4.0
```

## Summary

| Method | Copies data? | Works with fields? | Works with ndarrays? |
|--------|-------------|-------------------|---------------------|
| `to_numpy()` / `from_numpy()` (default) | yes | yes | yes |
| `to_torch()` / `from_torch()` (default) | yes | yes | yes |
| `to_numpy(copy=False)` / `to_torch(copy=False)` | no (DLPack view) | yes | yes |
| Direct pass-through | no | no | yes (as kernel arg) |

The `copy` parameter is supported on `to_numpy()` and `to_torch()` for `ScalarField`, `MatrixField` (and `VectorField`), `StructField`, and all `Ndarray` types. See [Zero-copy interop via DLPack](#zero-copy-interop-via-dlpack) for the support matrix and lifetime rules.
