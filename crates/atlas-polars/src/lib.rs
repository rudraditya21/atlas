//! Safe Polars interoperability for Atlas.
//!
//! This crate is the optional Polars integration boundary. It converts through Polars' public
//! Arrow chunk APIs and never exposes or requires unsafe code.

#![forbid(unsafe_code)]

use atlas_ndarray::{AtlasNdError, NDArray, OperandMetadata};
use polars::prelude::{Column, DataFrame, Series};
use polars_arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use thiserror::Error;

pub type AtlasPolarsResult<T> = Result<T, AtlasPolarsError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AtlasPolarsError {
    #[error("invalid input rank for {op}: expected {expected}, got rank {rank}")]
    InvalidInputRank { op: &'static str, expected: &'static str, rank: usize },
    #[error("column name count mismatch: expected {expected}, got {actual}")]
    ColumnNameCountMismatch { expected: usize, actual: usize },
    #[error("unsupported or mismatched dtype in column {column}")]
    ColumnDTypeMismatch { column: usize },
    #[error("null values are rejected for {op}")]
    NullValues { op: &'static str },
    #[error(transparent)]
    NdArray(#[from] AtlasNdError),
    #[error("Polars conversion failed: {0}")]
    Polars(String),
}

pub trait PolarsPrimitive: atlas_arrow::ArrowPrimitive {
    type PolarsArray: Array;
    fn to_polars_arrow(values: Vec<Self>) -> ArrayRef
    where
        Self: Sized;
    fn from_polars_arrow(
        array: &Self::PolarsArray,
        op: &'static str,
    ) -> AtlasPolarsResult<Vec<Self>>
    where
        Self: Sized;
}

macro_rules! impl_polars_primitive {
    ($($ty:ty => $array:ty),+ $(,)?) => {$(
        impl PolarsPrimitive for $ty {
            type PolarsArray = $array;
            fn to_polars_arrow(values: Vec<Self>) -> ArrayRef { Box::new(<$array>::from_vec(values)) }
            fn from_polars_arrow(array: &Self::PolarsArray, op: &'static str) -> AtlasPolarsResult<Vec<Self>> {
                if array.null_count() != 0 { return Err(AtlasPolarsError::NullValues { op }); }
                Ok(array.values_iter().copied().collect())
            }
        }
    )+};
}

impl_polars_primitive!(i8 => Int8Array, i16 => Int16Array, i32 => Int32Array, i64 => Int64Array,
    u8 => UInt8Array, u16 => UInt16Array, u32 => UInt32Array, u64 => UInt64Array,
    f32 => Float32Array, f64 => Float64Array);

impl PolarsPrimitive for bool {
    type PolarsArray = BooleanArray;
    fn to_polars_arrow(values: Vec<Self>) -> ArrayRef {
        Box::new(BooleanArray::from_slice(values))
    }
    fn from_polars_arrow(
        array: &Self::PolarsArray,
        op: &'static str,
    ) -> AtlasPolarsResult<Vec<Self>> {
        if array.null_count() != 0 {
            return Err(AtlasPolarsError::NullValues { op });
        }
        Ok(array.values_iter().collect())
    }
}

pub fn to_polars_series<T, O>(name: &str, array: &O) -> AtlasPolarsResult<Series>
where
    T: PolarsPrimitive,
    O: OperandMetadata<T> + ?Sized,
{
    if array.ndim() != 1 {
        return Err(AtlasPolarsError::InvalidInputRank {
            op: "to_polars_series",
            expected: "rank-1 vector",
            rank: array.ndim(),
        });
    }
    let values = (0..array.shape()[0])
        .map(|index| array.data()[array.offset() + index * array.strides()[0]])
        .collect();
    Series::from_arrow(name.into(), T::to_polars_arrow(values))
        .map_err(|error| AtlasPolarsError::Polars(error.to_string()))
}

pub fn from_polars_series<T: PolarsPrimitive>(series: &Series) -> AtlasPolarsResult<NDArray<T>> {
    let mut values = Vec::new();
    for array in series.chunks() {
        let array = array
            .as_any()
            .downcast_ref::<T::PolarsArray>()
            .ok_or(AtlasPolarsError::ColumnDTypeMismatch { column: 0 })?;
        values.extend(T::from_polars_arrow(array, "from_polars_series")?);
    }
    Ok(NDArray::from_shape_vec([series.len()], values)?)
}

pub fn to_polars_dataframe<T, O>(matrix: &O, names: &[&str]) -> AtlasPolarsResult<DataFrame>
where
    T: PolarsPrimitive,
    O: OperandMetadata<T> + ?Sized,
{
    if matrix.ndim() != 2 {
        return Err(AtlasPolarsError::InvalidInputRank {
            op: "to_polars_dataframe",
            expected: "rank-2 matrix",
            rank: matrix.ndim(),
        });
    }
    let [rows, columns]: [usize; 2] = matrix.shape().try_into().expect("rank was validated");
    if names.len() != columns {
        return Err(AtlasPolarsError::ColumnNameCountMismatch {
            expected: columns,
            actual: names.len(),
        });
    }
    let series = (0..columns)
        .map(|column| {
            to_polars_series(
                names[column],
                &NDArray::from_shape_vec(
                    [rows],
                    (0..rows)
                        .map(|row| {
                            matrix.data()[matrix.offset()
                                + row * matrix.strides()[0]
                                + column * matrix.strides()[1]]
                        })
                        .collect(),
                )?,
            )
            .map(Column::from)
        })
        .collect::<AtlasPolarsResult<Vec<_>>>()?;
    DataFrame::new(series).map_err(|error| AtlasPolarsError::Polars(error.to_string()))
}

pub fn from_polars_dataframe<T: PolarsPrimitive>(
    frame: &DataFrame,
) -> AtlasPolarsResult<NDArray<T>> {
    let columns = frame
        .get_columns()
        .iter()
        .enumerate()
        .map(|(index, column)| {
            from_polars_series::<T>(column.as_materialized_series()).map_err(|error| match error {
                AtlasPolarsError::ColumnDTypeMismatch { .. } => {
                    AtlasPolarsError::ColumnDTypeMismatch { column: index }
                }
                other => other,
            })
        })
        .collect::<AtlasPolarsResult<Vec<_>>>()?;
    let mut values = Vec::new();
    for row in 0..frame.height() {
        for column in &columns {
            values.push(column.data()[row]);
        }
    }
    Ok(NDArray::from_shape_vec([frame.height(), frame.width()], values)?)
}

#[cfg(test)]
mod tests {
    use atlas_arrow::{
        from_arrow_primitive, from_arrow_record_batch, to_arrow_primitive, to_arrow_record_batch,
    };
    use atlas_ndarray::{NDArray, SliceRange};
    use polars::prelude::DataType;

    use crate::{from_polars_dataframe, from_polars_series, to_polars_dataframe, to_polars_series};

    #[test]
    fn numeric_series_round_trip_preserves_dtype_and_values() {
        let values = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let series = to_polars_series("values", &values).unwrap();
        let converted = from_polars_series::<i32>(&series).unwrap();

        assert_eq!(series.name().as_str(), "values");
        assert_eq!(series.dtype(), &DataType::Int32);
        assert_eq!(converted.shape(), &[3]);
        assert_eq!(converted.data(), values.data());
    }

    #[test]
    fn series_conversion_uses_logical_view_values() {
        let values = NDArray::from_shape_vec([5], vec![0_i32, 1, 2, 3, 4]).unwrap();
        let offset = values.view().slice([1], [3]).unwrap();
        let strided = values.view().slice_ranges([SliceRange::new(Some(0), Some(5), 2)]).unwrap();
        let empty = values.view().slice([0], [0]).unwrap();

        let offset_series = to_polars_series("offset", &offset).unwrap();
        let strided_series = to_polars_series("strided", &strided).unwrap();
        let empty_series = to_polars_series("empty", &empty).unwrap();

        assert_eq!(offset_series.dtype(), &DataType::Int32);
        assert_eq!(from_polars_series::<i32>(&offset_series).unwrap().data(), &[1, 2, 3]);
        assert_eq!(from_polars_series::<i32>(&strided_series).unwrap().data(), &[0, 2, 4]);
        assert_eq!(empty_series.len(), 0);
        assert_eq!(empty_series.dtype(), &DataType::Int32);
    }

    #[test]
    fn numeric_dataframe_round_trip_preserves_column_orientation_and_dtype() {
        let matrix = NDArray::from_shape_vec([2, 3], vec![1_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let frame = to_polars_dataframe(&matrix, &["x", "y", "z"]).unwrap();
        let converted = from_polars_dataframe::<f64>(&frame).unwrap();

        assert_eq!(frame.shape(), (2, 3));
        assert_eq!(frame.column("y").unwrap().dtype(), &DataType::Float64);
        assert_eq!(converted.shape(), &[2, 3]);
        assert_eq!(converted.data(), matrix.data());
    }

    #[test]
    fn dataframe_conversion_uses_logical_matrix_view_values() {
        let matrix = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let transposed = matrix.view().transpose();
        let sliced = matrix.view().slice([0, 1], [2, 2]).unwrap();
        let empty = matrix.view().slice([0, 0], [0, 2]).unwrap();

        let transposed_frame = to_polars_dataframe(&transposed, &["left", "right"]).unwrap();
        let sliced_frame = to_polars_dataframe(&sliced, &["first", "second"]).unwrap();
        let empty_frame = to_polars_dataframe(&empty, &["first", "second"]).unwrap();

        assert_eq!(transposed_frame.shape(), (3, 2));
        assert_eq!(transposed_frame.get_column_names()[0].as_str(), "left");
        assert_eq!(transposed_frame.get_column_names()[1].as_str(), "right");
        assert_eq!(transposed_frame.column("left").unwrap().dtype(), &DataType::Int32);
        assert_eq!(
            from_polars_dataframe::<i32>(&transposed_frame).unwrap().data(),
            &[1, 4, 2, 5, 3, 6]
        );
        assert_eq!(from_polars_dataframe::<i32>(&sliced_frame).unwrap().data(), &[2, 3, 5, 6]);
        assert_eq!(empty_frame.shape(), (0, 2));
        assert_eq!(from_polars_dataframe::<i32>(&empty_frame).unwrap().shape(), &[0, 2]);
    }

    #[test]
    fn arrow_and_polars_conversions_match_equivalent_logical_arrays() {
        let vector_source = NDArray::from_shape_vec([5], vec![0_i32, 1, 2, 3, 4]).unwrap();
        let vector_view =
            vector_source.view().slice_ranges([SliceRange::new(Some(1), Some(5), 2)]).unwrap();
        let vector = NDArray::from_shape_vec([2], vec![1_i32, 3]).unwrap();

        assert_eq!(
            from_arrow_primitive::<i32>(&to_arrow_primitive(&vector_view).unwrap()).unwrap().data(),
            from_arrow_primitive::<i32>(&to_arrow_primitive(&vector).unwrap()).unwrap().data()
        );
        assert_eq!(
            from_polars_series::<i32>(&to_polars_series("values", &vector_view).unwrap())
                .unwrap()
                .data(),
            from_polars_series::<i32>(&to_polars_series("values", &vector).unwrap())
                .unwrap()
                .data()
        );

        let matrix_source = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let matrix_view = matrix_source.view().transpose();
        let matrix = NDArray::from_shape_vec([3, 2], vec![1_i32, 4, 2, 5, 3, 6]).unwrap();

        assert_eq!(
            from_arrow_record_batch::<i32>(
                &to_arrow_record_batch(&matrix_view, &["left", "right"]).unwrap(),
            )
            .unwrap()
            .data(),
            from_arrow_record_batch::<i32>(
                &to_arrow_record_batch(&matrix, &["left", "right"]).unwrap(),
            )
            .unwrap()
            .data()
        );
        assert_eq!(
            from_polars_dataframe::<i32>(
                &to_polars_dataframe(&matrix_view, &["left", "right"]).unwrap(),
            )
            .unwrap()
            .data(),
            from_polars_dataframe::<i32>(
                &to_polars_dataframe(&matrix, &["left", "right"]).unwrap(),
            )
            .unwrap()
            .data()
        );
    }
}
