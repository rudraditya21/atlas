use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::{Field, Schema};
use atlas_ndarray::NDArray;

use crate::{ArrowPrimitive, AtlasArrowError, AtlasArrowResult};

/// Converts a rank-2 Atlas matrix into an Arrow record batch.
///
/// Atlas rows become record-batch rows; each Atlas column becomes one non-nullable Arrow column.
/// Values are copied into independent Arrow buffers.
pub fn to_arrow_record_batch<T: ArrowPrimitive>(
    matrix: &NDArray<T>,
    column_names: &[&str],
) -> AtlasArrowResult<RecordBatch> {
    if matrix.ndim() != 2 {
        return Err(AtlasArrowError::InvalidInputRank {
            op: "to_arrow_record_batch",
            expected: "rank-2 matrix",
            rank: matrix.ndim(),
        });
    }
    let rows = matrix.shape()[0];
    let columns = matrix.shape()[1];
    if column_names.len() != columns {
        return Err(AtlasArrowError::ColumnNameCountMismatch {
            op: "to_arrow_record_batch",
            expected: columns,
            actual: column_names.len(),
        });
    }
    let fields: Vec<_> =
        column_names.iter().map(|name| Field::new(*name, T::ARROW_DATA_TYPE, false)).collect();
    let arrays = (0..columns)
        .map(|column| {
            let values = (0..rows).map(|row| matrix.data()[row * columns + column]).collect();
            T::to_arrow_ref(values)
        })
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|error| AtlasArrowError::RecordBatch { reason: error.to_string() })
}

#[cfg(test)]
mod tests {
    use arrow_array::{Array, Int32Array};
    use atlas_ndarray::NDArray;

    use crate::{AtlasArrowError, to_arrow_record_batch};

    #[test]
    fn record_batch_uses_matrix_columns_as_arrow_columns() {
        let matrix = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let batch = to_arrow_record_batch(&matrix, &["a", "b", "c"]).unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(batch.schema().field(1).name(), "b");
        assert_eq!(
            batch.column(1).as_any().downcast_ref::<Int32Array>().unwrap().values(),
            &[2, 5]
        );
    }

    #[test]
    fn record_batch_validates_rank_and_column_names() {
        let vector = NDArray::from_shape_vec([2], vec![1_i32, 2]).unwrap();
        let matrix = NDArray::from_shape_vec([1, 2], vec![1_i32, 2]).unwrap();

        assert_eq!(
            to_arrow_record_batch(&vector, &[]).unwrap_err(),
            AtlasArrowError::InvalidInputRank {
                op: "to_arrow_record_batch",
                expected: "rank-2 matrix",
                rank: 1,
            }
        );
        assert_eq!(
            to_arrow_record_batch(&matrix, &["a"]).unwrap_err(),
            AtlasArrowError::ColumnNameCountMismatch {
                op: "to_arrow_record_batch",
                expected: 2,
                actual: 1,
            }
        );
    }
}
