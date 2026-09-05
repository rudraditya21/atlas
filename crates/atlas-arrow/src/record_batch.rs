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

/// Converts a homogeneous Arrow record batch into a newly allocated rank-2 Atlas matrix.
///
/// Each record-batch row becomes one Atlas row. Every column must have the Arrow dtype for `T`
/// and contain no nulls; values are copied into dense row-major Atlas storage.
pub fn from_arrow_record_batch<T: ArrowPrimitive>(
    batch: &RecordBatch,
) -> AtlasArrowResult<NDArray<T>> {
    let columns = batch
        .columns()
        .iter()
        .enumerate()
        .map(|(index, array)| {
            let column = array.as_any().downcast_ref::<T::Array>().ok_or_else(|| {
                AtlasArrowError::ColumnDTypeMismatch {
                    op: "from_arrow_record_batch",
                    column: index,
                    expected: T::ARROW_DATA_TYPE.to_string(),
                    actual: array.data_type().to_string(),
                }
            })?;
            T::from_arrow(column, "from_arrow_record_batch")
        })
        .collect::<AtlasArrowResult<Vec<_>>>()?;
    let mut values = Vec::new();
    for row in 0..batch.num_rows() {
        for column in &columns {
            values.push(column[row]);
        }
    }
    Ok(NDArray::from_shape_vec([batch.num_rows(), batch.num_columns()], values)?)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use atlas_ndarray::NDArray;

    use crate::{AtlasArrowError, from_arrow_record_batch, to_arrow_record_batch};

    #[test]
    fn record_batch_uses_matrix_columns_as_arrow_columns() {
        let matrix = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let batch = to_arrow_record_batch(&matrix, &["a", "b", "c"]).unwrap();

        assert_eq!(batch.num_rows(), 2);
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(batch.schema().field(1).name(), "b");
        assert!(!batch.schema().field(1).is_nullable());
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

    #[test]
    fn reverse_record_batch_conversion_preserves_row_major_order() {
        let matrix = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let batch = to_arrow_record_batch(&matrix, &["a", "b", "c"]).unwrap();
        let converted = from_arrow_record_batch::<i32>(&batch).unwrap();

        assert_eq!(converted.shape(), matrix.shape());
        assert_eq!(converted.data(), matrix.data());
    }

    #[test]
    fn reverse_record_batch_conversion_rejects_mismatched_dtypes_and_nulls() {
        let mismatched =
            to_arrow_record_batch(&NDArray::from_shape_vec([1, 1], vec![1_i64]).unwrap(), &["a"])
                .unwrap();
        let nullable = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)])),
            vec![Arc::new(Int32Array::from(vec![Some(1_i32), None])) as ArrayRef],
        )
        .unwrap();

        assert_eq!(
            from_arrow_record_batch::<i32>(&mismatched).unwrap_err(),
            AtlasArrowError::ColumnDTypeMismatch {
                op: "from_arrow_record_batch",
                column: 0,
                expected: "Int32".to_owned(),
                actual: "Int64".to_owned(),
            }
        );
        assert_eq!(
            from_arrow_record_batch::<i32>(&nullable).unwrap_err(),
            AtlasArrowError::NullValues { op: "from_arrow_record_batch" }
        );
    }

    #[test]
    fn reverse_record_batch_conversion_respects_offsets_and_empty_batches() {
        let values = Int32Array::from(vec![1_i32, 2, 3, 4]).slice(1, 2);
        let offset_batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)])),
            vec![Arc::new(values) as ArrayRef],
        )
        .unwrap();
        let empty = NDArray::from_shape_vec([0, 2], Vec::<i32>::new()).unwrap();
        let empty_batch = to_arrow_record_batch(&empty, &["a", "b"]).unwrap();

        let converted = from_arrow_record_batch::<i32>(&offset_batch).unwrap();
        let empty_converted = from_arrow_record_batch::<i32>(&empty_batch).unwrap();

        assert_eq!(converted.shape(), &[2, 1]);
        assert_eq!(converted.data(), &[2, 3]);
        assert_eq!(empty_converted.shape(), &[0, 2]);
        assert!(empty_converted.data().is_empty());
    }

    #[test]
    fn record_batch_chunks_convert_independently() {
        let first =
            to_arrow_record_batch(&NDArray::from_shape_vec([1, 1], vec![1_i32]).unwrap(), &["a"])
                .unwrap();
        let second =
            to_arrow_record_batch(&NDArray::from_shape_vec([1, 1], vec![2_i32]).unwrap(), &["a"])
                .unwrap();

        assert_eq!(from_arrow_record_batch::<i32>(&first).unwrap().data(), &[1]);
        assert_eq!(from_arrow_record_batch::<i32>(&second).unwrap().data(), &[2]);
    }

    #[test]
    fn reverse_record_batch_conversion_rejects_unsupported_dtypes() {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("a", DataType::Utf8, false)])),
            vec![Arc::new(StringArray::from(vec!["atlas"])) as ArrayRef],
        )
        .unwrap();

        assert_eq!(
            from_arrow_record_batch::<i32>(&batch).unwrap_err(),
            AtlasArrowError::ColumnDTypeMismatch {
                op: "from_arrow_record_batch",
                column: 0,
                expected: DataType::Int32.to_string(),
                actual: DataType::Utf8.to_string(),
            }
        );
    }
}
