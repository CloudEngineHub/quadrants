import dataclasses
import enum
import numbers
import time
from typing import Any, Sequence

import numpy as np

from quadrants import _logging, _tensor_wrapper
from quadrants._tensor_wrapper import _TENSOR_WRAPPER_TYPES
from quadrants._tensor_wrapper import Tensor as _TensorWrapper
from quadrants.types.annotations import Template

from .._ndarray import ScalarNdarray
from .._quadrants_callable import BoundQuadrantsCallable, QuadrantsCallable
from ..field import ScalarField
from ..kernel_arguments import ArgMetadata
from ..matrix import MatrixField, MatrixNdarray, VectorNdarray
from ..util import is_data_oriented, is_dataclass_instance
from .hash_utils import hash_iterable_strings

_FIELD_TYPES = (ScalarField, MatrixField)

try:
    import torch

    torch_type = torch.Tensor
except ImportError:
    torch_type = ()


g_num_calls = 0
g_num_args = 0
g_hashing_time = 0
g_repr_time = 0
g_num_ignored_calls = 0


FIELD_METADATA_CACHE_VALUE = "add_value_to_cache_key"

_DC_REPR_NONE = object()


# Sentinel returned by ``stringify_obj_type`` when a recognised-but-unsupported tensor-like type (``Field`` /
# ``MatrixField``) is encountered anywhere in the traversal. Containers that see this sentinel (``dataclass_to_repr``,
# the ``data_oriented`` branch, and the top-level ``hash_args`` loop) must propagate it upward — fastcache cannot
# safely hash the call. Distinct from ``None``, which means "opaque type, safe to silently skip at nested levels".
class _FailFastcache:
    """Singleton sentinel; identity-compared. See module docstring on ``stringify_obj_type``'s return contract."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


_FAIL_FASTCACHE = _FailFastcache()


class FastcacheSkip(enum.Enum):
    """Why fastcache does not apply to this call."""

    FIELD_VIA_TENSOR = "field_via_tensor"
    WARN = "warn"


# Set when the fastcache skip is something callers should warn about (as opposed to a Field arriving through a
# qd.Tensor annotation, which is a normal silent path). Reset at the start of each hash_args call.
_should_warn = False


def _mark_warn_if_not_tensor_annotation(arg_meta: ArgMetadata | None) -> None:
    """Flag that a warning is needed if the Field didn't arrive through a qd.Tensor annotation."""
    global _should_warn  # pylint: disable=global-statement
    if arg_meta is not None and arg_meta.annotation is not _TensorWrapper:
        _should_warn = True


def _mark_should_warn() -> None:
    global _should_warn  # pylint: disable=global-statement
    _should_warn = True


def dataclass_to_repr(raise_on_templated_floats: bool, path: tuple[str, ...], arg: Any) -> str | _FailFastcache | None:
    """Hash a dataclass instance.

    Returns:
      - ``str``: a string representation suitable for the fastcache key.
      - ``_FAIL_FASTCACHE``: a recognised-but-unsupported tensor-like field was hit; fastcache must be disabled
        for the whole call.
      - ``None``: dataclass-level skip. Currently unused (dataclasses always succeed unless they hit Field/MatrixField),
        but defined symmetrically with ``stringify_obj_type``.

    Note that opaque-typed fields (UUID, plain Python objects, ...) are *silently skipped* — they cannot affect
    kernel codegen because the kernel cannot read non-recognised Python types, so omitting them from the hash is
    safe by construction.
    """
    # PERF: For frozen dataclasses, the repr never changes. Cache it on the instance to avoid repeated
    # ``dataclasses.fields()`` calls (which are slow due to extra runtime checks — see _template_mapper_hotpath.py
    # module docstring). The cache is stored as ``_qd_dc_repr`` via ``object.__setattr__`` to bypass frozen guards.
    # A cached ``_DC_REPR_NONE`` is stored to distinguish "not yet computed" from "computed but not fast-cacheable".
    is_frozen = type(arg).__hash__ is not None
    if is_frozen:
        cached = getattr(arg, "_qd_dc_repr", None)
        if cached is _DC_REPR_NONE:
            return _FAIL_FASTCACHE
        if cached is not None:
            return cached
    repr_l = []
    for field in dataclasses.fields(arg):
        child_value = getattr(arg, field.name)
        _repr = stringify_obj_type(raise_on_templated_floats, path + (field.name,), child_value, arg_meta=None)
        if _repr is _FAIL_FASTCACHE:
            # Recognised-but-unsupported (Field/MatrixField) somewhere in this child's subtree. Mark whether the
            # field arrived via a non-Tensor annotation so the top-level decides between WARN and FIELD_VIA_TENSOR.
            if isinstance(child_value, _FIELD_TYPES) and field.type is not _TensorWrapper:
                _mark_should_warn()
            if is_frozen:
                try:
                    object.__setattr__(arg, "_qd_dc_repr", _DC_REPR_NONE)
                except AttributeError:
                    pass
            return _FAIL_FASTCACHE
        if _repr is None:
            # Opaque-typed field; skip silently. Opaque types cannot affect kernel codegen because the kernel
            # cannot read non-recognised Python types — they are inert metadata.
            continue
        full_repr = f"{field.name}: ({_repr})"
        if field.metadata.get(FIELD_METADATA_CACHE_VALUE, False):
            full_repr += f" = {child_value}"
        repr_l.append(full_repr)
    result = "[" + ",".join(repr_l) + "]"
    if is_frozen:
        try:
            object.__setattr__(arg, "_qd_dc_repr", result)
        except AttributeError:
            pass
    return result


def _is_template(arg_meta: ArgMetadata | None) -> bool:
    if arg_meta is None:
        return False
    annot = arg_meta.annotation
    return annot is Template or isinstance(annot, Template)


def stringify_obj_type(
    raise_on_templated_floats: bool,
    path: tuple[str, ...],
    obj: object,
    arg_meta: ArgMetadata | None,
    nested: bool = False,
) -> str | _FailFastcache | None:
    """
    Convert an object into a string representation that only depends on its type (and, where relevant, its value).

    Return contract:
      - ``str``: the object is hashable for fastcache; the returned string contributes to the cache key.
      - ``_FAIL_FASTCACHE``: a recognised-but-unsupported type (``qd.field`` / ``qd.Matrix.field``) was encountered.
        Containers must propagate this upward; fastcache will be disabled for the whole call.
      - ``None``: the object's type is *opaque* — not recognised by the hasher. Containers (``dataclass_to_repr``
        and the ``data_oriented`` branch below) silently skip opaque members because opaque types cannot affect
        kernel codegen (the kernel can only read recognised types: ndarrays, primitives, enums, dataclasses,
        nested ``@qd.data_oriented`` objects). At the top level (``nested=False``), opaque is treated as
        an error and a ``[FASTCACHE][PARAM_INVALID]`` warning is emitted.

    Parameters:
      - ``nested``: ``True`` if this call comes from a container walker (dataclass / data_oriented). Suppresses
        the top-level ``[FASTCACHE][PARAM_INVALID]`` warning for opaque types so nested opaque members are
        skipped silently. ``False`` at the top of each kernel-arg traversal.
      - ``arg_meta``: non-``None`` only for the top-level kernel arguments and for ``@qd.data_oriented`` members.
        Used to determine whether to bake values into the cache key (primitives in template positions, and all
        primitive members of data-oriented containers).
    """
    # ``qd.Tensor`` wrappers passed as struct fields. The top-level kernel-arg unwrap hook in ``Kernel.__call__`` strips
    # wrappers off positional / keyword args before the fastcache hasher sees them, but the dataclass / data-oriented
    # walkers below (``dataclass_to_repr`` and the ``is_data_oriented`` branch) do raw ``getattr`` to fetch struct
    # fields, so a wrapper stored as a struct field arrives here un-stripped. Without this branch the hasher falls
    # through to the ``[FASTCACHE][PARAM_INVALID]`` warning and disables the fast path for the whole call. See
    # ``perso_hugh/doc/quadrants-tensor.md`` §8.14.
    #
    # PERF-CRITICAL: The _any_tensor_constructed guard makes this check zero-cost when no qd.Tensor has been created.
    # ``type(obj) in _TENSOR_WRAPPER_TYPES`` is used instead of ``isinstance`` because it is a pointer comparison (~10
    # ns) vs an MRO walk (~100–200 ns). Do not replace with isinstance or remove the guard.
    if (
        _tensor_wrapper._any_tensor_constructed and type(obj) in _TENSOR_WRAPPER_TYPES
    ):  # pyright: ignore[reportOptionalMemberAccess]
        obj = obj._unwrap()  # pyright: ignore[reportAttributeAccessIssue]
    arg_type = type(obj)
    _layout = getattr(obj, "_qd_layout", None)
    _layout_tag = "" if _layout is None else f"-L{_layout!r}"
    if isinstance(obj, ScalarNdarray):
        return f"[nd-{obj.dtype}-{len(obj.shape)}{_layout_tag}]"  # type: ignore[arg-type]
    if isinstance(obj, VectorNdarray):
        return f"[ndv-{obj.n}-{obj.dtype}-{len(obj.shape)}{_layout_tag}]"  # type: ignore[arg-type]
    if isinstance(obj, ScalarField):
        # Recognised-but-unsupported: Field's shape/dtype affect kernel codegen but fastcache doesn't yet know how
        # to handle them. Disable fastcache for the whole call.
        # TODO: think about whether there is a way to include fields
        _mark_warn_if_not_tensor_annotation(arg_meta)
        return _FAIL_FASTCACHE
    if isinstance(obj, MatrixNdarray):
        return f"[ndm-{obj.m}-{obj.n}-{obj.dtype}-{len(obj.shape)}{_layout_tag}]"  # type: ignore[arg-type]
    if isinstance(obj, torch_type):
        return f"[pt-{obj.dtype}-{obj.ndim}]"  # type: ignore
    if isinstance(obj, np.ndarray):
        return f"[np-{obj.dtype}-{obj.ndim}]"
    if isinstance(obj, MatrixField):
        # Recognised-but-unsupported, same as ScalarField above.
        # TODO: think about whether there is a way to include fields
        _mark_warn_if_not_tensor_annotation(arg_meta)
        return _FAIL_FASTCACHE
    if is_dataclass_instance(obj):
        return dataclass_to_repr(raise_on_templated_floats, path, obj)
    if is_data_oriented(obj):
        # Walk the data_oriented container's members. Recognised members contribute to the cache key; recognised-
        # but-unsupported (Field/MatrixField) propagates _FAIL_FASTCACHE; opaque-typed members are skipped silently.
        #
        # Silently skipping opaque members is safe by construction: the kernel can only read recognised member types
        # (ndarrays, primitives, enums, dataclasses, nested data_oriented). Opaque Python objects (UUIDs, Pydantic
        # ``BaseModel`` instances, back-pointers up the object graph, etc.) cannot be read by kernel code, so they
        # cannot affect kernel codegen and omitting them from the hash is correct.
        child_repr_l = ["da"]
        try:
            _asdict = getattr(obj, "_asdict")
            _dict = _asdict()
        except AttributeError:
            _dict = obj.__dict__
        for k, v in _dict.items():
            # Skip Quadrants method-descriptor cache entries. ``QuadrantsCallable.__get__`` stashes the per-instance
            # ``BoundQuadrantsCallable`` on ``instance.__dict__`` so that subsequent ``instance.method`` lookups skip
            # the descriptor allocation; those entries are not data and must not invalidate the fastcache key.
            v_type = type(v)
            if v_type is QuadrantsCallable or v_type is BoundQuadrantsCallable:
                continue
            _child_repr = stringify_obj_type(
                raise_on_templated_floats, (*path, k), v, ArgMetadata(Template, ""), nested=True
            )
            if _child_repr is _FAIL_FASTCACHE:
                return _FAIL_FASTCACHE
            if _child_repr is None:
                # Opaque member; skip silently.
                continue
            child_repr_l.append(f"{k}: {_child_repr}")
        return ", ".join(child_repr_l)
    if issubclass(arg_type, (numbers.Number, np.number)):
        if _is_template(arg_meta):
            if raise_on_templated_floats and isinstance(obj, float):
                raise ValueError("Floats should not be used in template parameters.")
            # cache value too
            return f"{arg_type}={obj}"
        return str(arg_type)
    if arg_type is np.bool_:
        # np is deprecating bool. Treat specially/carefully
        if _is_template(arg_meta):
            # cache value too
            return f"np.bool_={obj}"
        return "np.bool_"
    if isinstance(obj, enum.Enum):
        return f"enum-{obj.name}-{obj.value}"
    # Opaque (unrecognised) type. At nested levels, container walkers skip these silently — opaque types cannot
    # affect kernel codegen because the kernel cannot read non-recognised Python types. At the top level, this is
    # a user error (the kernel's argument is uninterpretable to fastcache) and we emit a warning.
    if nested:
        return None
    _mark_should_warn()
    # The bit in caps should not be modified without updating corresponding test
    # The rest of free text can be freely modified
    # (will probably formalize this in more general doc / contributor guidelines at some point)
    _logging.warn(
        f"[FASTCACHE][PARAM_INVALID] Parameter with path {path} and type {arg_type} not allowed by fast cache."
    )
    return None


def hash_args(
    raise_on_templated_floats: bool, args: Sequence[Any], arg_metas: Sequence[ArgMetadata | None]
) -> str | FastcacheSkip:
    """Return the args hash string, or a FastcacheSkip explaining why hashing failed."""
    global g_num_calls, g_num_args, g_hashing_time, g_repr_time, g_num_ignored_calls, _should_warn  # pylint: disable=global-statement
    _should_warn = False
    g_num_calls += 1
    g_num_args += len(args)
    hash_l = []
    if len(args) != len(arg_metas):
        raise RuntimeError(
            f"Number of args passed in {len(args)} doesnt match number of declared args {len(arg_metas)}"
        )
    for i_arg, arg in enumerate(args):
        start = time.time()
        _hash = stringify_obj_type(raise_on_templated_floats, (str(i_arg),), arg, arg_metas[i_arg], nested=False)
        g_repr_time += time.time() - start
        # Both ``_FAIL_FASTCACHE`` (recognised-but-unsupported) and ``None`` (opaque at top level) disable
        # fastcache. ``_should_warn`` selects between WARN (loud) and FIELD_VIA_TENSOR (silent — Field reached via
        # qd.Tensor annotation, which is a normal path).
        if _hash is _FAIL_FASTCACHE or _hash is None or not _hash:
            g_num_ignored_calls += 1
            return FastcacheSkip.WARN if _should_warn else FastcacheSkip.FIELD_VIA_TENSOR
        hash_l.append(_hash)
    start = time.time()
    res = hash_iterable_strings(hash_l)
    g_hashing_time += time.time() - start
    return res


def dump_stats() -> None:
    print("args hasher dump stats")
    print("total calls", g_num_calls)
    print("ignored calls", g_num_ignored_calls)
    print("total args", g_num_args)
    print("hashing time", g_hashing_time)
    print("arg representation time", g_repr_time)
