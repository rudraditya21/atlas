use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::DataType;
use atlas_ndarray::NDArray;

use crate::{AtlasArrowError, AtlasArrowResult, InterchangeDType};

/// Converts a rank-1 Atlas array into a newly allocated Arrow primitive array.
///
/// The conversion always copies values: Atlas retains ownership of its storage and the returned
/// Arrow array owns an independent buffer. Platform-sized integers are intentionally excluded
/// because Arrow interchange requires a fixed-width integer representation.
pub fn to_arrow_primitive<T: ArrowPrimitive>(array: &NDArray<T>) -> AtlasArrowResult<T::Array> {
    if array.ndim() != 1 {
        return Err(AtlasArrowError::InvalidInputRank {
            op: "to_arrow_primitive",
            expected: "rank-1 vector",
            rank: array.ndim(),
        });
    }

    Ok(T::to_arrow(array.data().to_vec()))
}

/// Converts an Arrow primitive array into a newly allocated rank-1 Atlas array.
///
/// Arrow offsets are respected through the array's logical iterator. Arrays containing nulls are
/// rejected because Atlas ndarrays do not carry a validity bitmap.
pub fn from_arrow_primitive<T: ArrowPrimitive>(array: &T::Array) -> AtlasArrowResult<NDArray<T>> {
    Ok(NDArray::from_shape_vec([array.len()], T::from_arrow(array, "from_arrow_primitive")?)?)
}

/// Maps a fixed-width Atlas primitive dtype to its Arrow primitive-array representation.
pub trait ArrowPrimitive: InterchangeDType {
    /// Arrow primitive array produced by this conversion.
    type Array: Array + 'static;

    #[doc(hidden)]
    const ARROW_DATA_TYPE: DataType;

    #[doc(hidden)]
    fn to_arrow(values: Vec<Self>) -> Self::Array
    where
        Self: Sized;

    #[doc(hidden)]
    fn from_arrow(array: &Self::Array, op: &'static str) -> AtlasArrowResult<Vec<Self>>
    where
        Self: Sized;

    #[doc(hidden)]
    fn to_arrow_ref(values: Vec<Self>) -> ArrayRef
    where
        Self: Sized;
}

macro_rules! impl_arrow_primitive {
    ($($ty:ty => $array:ty, $data_type:expr),+ $(,)?) => {
        $(
            impl ArrowPrimitive for $ty {
                type Array = $array;

                const ARROW_DATA_TYPE: DataType = $data_type;

                fn to_arrow(values: Vec<Self>) -> Self::Array { values.into() }

                fn from_arrow(array: &Self::Array, op: &'static str) -> AtlasArrowResult<Vec<Self>> {
                    if array.null_count() != 0 {
                        return Err(AtlasArrowError::NullValues { op });
                    }
                    Ok(array.iter().map(|value| value.expect("null count was checked")).collect())
                }

                fn to_arrow_ref(values: Vec<Self>) -> ArrayRef { Arc::new(Self::to_arrow(values)) }
            }
        )+
    };
}

impl_arrow_primitive!(
    bool => BooleanArray, DataType::Boolean,
    i8 => Int8Array, DataType::Int8,
    i16 => Int16Array, DataType::Int16,
    i32 => Int32Array, DataType::Int32,
    i64 => Int64Array, DataType::Int64,
    u8 => UInt8Array, DataType::UInt8,
    u16 => UInt16Array, DataType::UInt16,
    u32 => UInt32Array, DataType::UInt32,
    u64 => UInt64Array, DataType::UInt64,
    f32 => Float32Array, DataType::Float32,
    f64 => Float64Array, DataType::Float64,
);

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use arrow_array::{Array, Int32Array};

    use crate::{AtlasArrowError, from_arrow_primitive, to_arrow_primitive};

    #[test]
    fn primitive_conversion_copies_rank_one_values() {
        let values = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let converted = to_arrow_primitive(&values).unwrap();

        assert_eq!(converted.values(), &[1, 2, 3]);
        assert_eq!(converted.null_count(), 0);
    }

    #[test]
    fn primitive_conversion_rejects_non_vector_inputs() {
        let values = NDArray::from_shape_vec([1, 2], vec![true, false]).unwrap();

        assert_eq!(
            to_arrow_primitive(&values).unwrap_err(),
            AtlasArrowError::InvalidInputRank {
                op: "to_arrow_primitive",
                expected: "rank-1 vector",
                rank: 2,
            }
        );
    }

    #[test]
    fn reverse_conversion_respects_arrow_offsets_and_rejects_nulls() {
        let values = Int32Array::from(vec![1_i32, 2, 3, 4]);
        let sliced = values.slice(1, 2);
        let sliced = sliced.as_any().downcast_ref::<Int32Array>().unwrap();
        let nullable = Int32Array::from(vec![Some(1_i32), None]);

        assert_eq!(from_arrow_primitive::<i32>(sliced).unwrap().data(), &[2, 3]);
        assert_eq!(
            from_arrow_primitive::<i32>(&nullable).unwrap_err(),
            AtlasArrowError::NullValues { op: "from_arrow_primitive" }
        );
    }

    #[test]
    fn primitive_conversion_preserves_empty_arrays() {
        let values = NDArray::from_shape_vec([0], Vec::<i32>::new()).unwrap();
        let arrow = to_arrow_primitive(&values).unwrap();
        let converted = from_arrow_primitive::<i32>(&arrow).unwrap();

        assert!(arrow.is_empty());
        assert_eq!(converted.shape(), &[0]);
        assert!(converted.data().is_empty());
    }
}
