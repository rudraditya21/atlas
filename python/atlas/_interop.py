"""Arrow interoperability bindings exposed by Atlas."""

from . import _native
from ._support import _coerce_arrays

to_arrow_primitive = _coerce_arrays(
    _native.to_arrow_primitive, required=((0, "value"),)
)
from_arrow_primitive = _native.from_arrow_primitive
to_arrow_record_batch = _coerce_arrays(
    _native.to_arrow_record_batch, required=((0, "matrix"),)
)
from_arrow_record_batch = _native.from_arrow_record_batch
