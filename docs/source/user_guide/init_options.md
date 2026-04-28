# qd.init options

`qd.init(...)` accepts every field of the underlying `CompileConfig` struct as a keyword argument; the same fields are also reachable as environment variables of the form `QD_<UPPERCASE_NAME>` (e.g. `QD_OFFLINE_CACHE=0`). This page covers some of the knobs that are commonly tuned in practice. The underlying source of truth is `quadrants/program/compile_config.h`.

## Compile-time tuning

### `cfg_optimization`

Whether to run the control-flow-graph optimization pass. Default `True`. Setting it to `False` makes compilation up to 6x faster while costing 1-5% of runtime speed; consider disabling it if compile time is the bottleneck and the runtime delta is acceptable.

### `fast_math`

Whether to enable IEEE-relaxed floating-point optimizations (FMA fusion, no NaN / infinity / signed-zero guarantees). Default `True`. Disable when investigating numerical anomalies or running deterministic-tolerance tests.

### `num_compile_threads`

Number of host threads used when compiling kernels. Default `4`. Raise on machines with many idle cores compiling many kernels back-to-back; lower (or set to `1`) on memory-pressure-bound systems where concurrent LLVM compilations thrash.

## Debugging

### `debug`

Enables IR verification between every compiler pass plus additional runtime checks (integer-overflow guards on arithmetic, linear-index overflow guards on tensor indexing, adstack push-bounds at the runtime helper level). Default `False`. Compile time slows substantially because the verifier walks the IR after every transform and the extra runtime checks expand the emitted code; ~21s additional has been observed on adstack-heavy kernels. Turn this on while iterating on a kernel that is producing incorrect numerics or while developing a new compiler pass; turn it back off once the bug is found.

### `check_out_of_bound`

Enables runtime bounds-checking for tensor indexing. Default `False`. Costs runtime performance proportional to indexing density; leave off for benchmarks. Backends that do not expose the `assertion` extension (currently Metal and Vulkan) cannot honor this flag.
