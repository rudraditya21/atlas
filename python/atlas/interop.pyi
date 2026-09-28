from typing import TYPE_CHECKING, Any
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import pyarrow as pa

Array = NDArray[Any]

def to_arrow_primitive(value: ArrayLike) -> pa.Array: ...
def from_arrow_primitive(value: pa.Array) -> Array: ...
def to_arrow_record_batch(
    matrix: ArrayLike, column_names: list[str]
) -> pa.RecordBatch: ...
def from_arrow_record_batch(value: pa.RecordBatch) -> Array: ...
