//! Safe Polars interoperability for Atlas.
//!
//! This crate is the optional Polars integration boundary. It converts through Polars' public
//! Arrow chunk APIs and never exposes or requires unsafe code.

#![forbid(unsafe_code)]

use atlas_ndarray::{AtlasNdError, NDArray};
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

pub fn to_polars_series<T: PolarsPrimitive>(
    name: &str,
    array: &NDArray<T>,
) -> AtlasPolarsResult<Series> {
    if array.ndim() != 1 {
        return Err(AtlasPolarsError::InvalidInputRank {
            op: "to_polars_series",
            expected: "rank-1 vector",
            rank: array.ndim(),
        });
    }
    Series::from_arrow(name.into(), T::to_polars_arrow(array.data().to_vec()))
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

pub fn to_polars_dataframe<T: PolarsPrimitive>(
    matrix: &NDArray<T>,
    names: &[&str],
) -> AtlasPolarsResult<DataFrame> {
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
                    (0..rows).map(|row| matrix.data()[row * columns + column]).collect(),
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
