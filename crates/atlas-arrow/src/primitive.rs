use arrow_array::{
    BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
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
            rank: array.ndim(),
        });
    }

    Ok(T::to_arrow(array.data().to_vec()))
}

/// Maps a fixed-width Atlas primitive dtype to its Arrow primitive-array representation.
pub trait ArrowPrimitive: InterchangeDType {
    /// Arrow primitive array produced by this conversion.
    type Array;

    #[doc(hidden)]
    fn to_arrow(values: Vec<Self>) -> Self::Array
    where
        Self: Sized;
}

macro_rules! impl_arrow_primitive {
    ($($ty:ty => $array:ty),+ $(,)?) => {
        $(
            impl ArrowPrimitive for $ty {
                type Array = $array;

                fn to_arrow(values: Vec<Self>) -> Self::Array { values.into() }
            }
        )+
    };
}

impl_arrow_primitive!(
    bool => BooleanArray,
    i8 => Int8Array,
    i16 => Int16Array,
    i32 => Int32Array,
    i64 => Int64Array,
    u8 => UInt8Array,
    u16 => UInt16Array,
    u32 => UInt32Array,
    u64 => UInt64Array,
    f32 => Float32Array,
    f64 => Float64Array,
);

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasArrowError, to_arrow_primitive};

    #[test]
    fn primitive_conversion_copies_rank_one_values() {
        let values = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let converted = to_arrow_primitive(&values).unwrap();

        assert_eq!(converted.values(), &[1, 2, 3]);
    }

    #[test]
    fn primitive_conversion_rejects_non_vector_inputs() {
        let values = NDArray::from_shape_vec([1, 2], vec![true, false]).unwrap();

        assert_eq!(
            to_arrow_primitive(&values).unwrap_err(),
            AtlasArrowError::InvalidInputRank { op: "to_arrow_primitive", rank: 2 }
        );
    }
}
