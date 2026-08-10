use crate::{AtlasNdError, AtlasNdResult};

pub trait AxisIndex: Copy {
    fn into_i64(self) -> i64;

    fn try_into_i64(self) -> Option<i64> {
        Some(self.into_i64())
    }
}

macro_rules! impl_signed_axis_index {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl AxisIndex for $ty {
                fn into_i64(self) -> i64 {
                    self as i64
                }

                fn try_into_i64(self) -> Option<i64> {
                    Some(self as i64)
                }
            }
        )+
    };
}

macro_rules! impl_unsigned_axis_index {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl AxisIndex for $ty {
                fn into_i64(self) -> i64 {
                    self as i64
                }

                fn try_into_i64(self) -> Option<i64> {
                    i64::try_from(self).ok()
                }
            }
        )+
    };
}

impl_signed_axis_index!(i8, i16, i32, i64, isize);
impl_unsigned_axis_index!(u8, u16, u32, u64, usize);

pub(crate) fn normalize_axis<A: AxisIndex>(axis: A, ndim: usize) -> AtlasNdResult<usize> {
    let axis = axis.try_into_i64().ok_or(AtlasNdError::InvalidAxis { axis: i64::MAX, ndim })?;
    let ndim_i64 = ndim as i64;

    let normalized = if axis < 0 { ndim_i64 + axis } else { axis };

    if normalized < 0 || normalized >= ndim_i64 {
        return Err(AtlasNdError::InvalidAxis { axis, ndim });
    }

    Ok(normalized as usize)
}

pub(crate) fn normalize_scalar_index<A: AxisIndex>(
    index: A,
    axis: usize,
    dim: usize,
) -> AtlasNdResult<usize> {
    let index = index.try_into_i64().ok_or(AtlasNdError::IndexOutOfBounds {
        axis,
        index: i64::MAX,
        dim,
    })?;
    let dim_i128 = dim as i128;
    let normalized = if index < 0 { dim_i128 + index as i128 } else { index as i128 };

    if normalized < 0 || normalized >= dim_i128 {
        return Err(AtlasNdError::IndexOutOfBounds { axis, index, dim });
    }

    Ok(normalized as usize)
}

pub(crate) fn normalize_and_offset_indices<A: AxisIndex>(
    base_offset: usize,
    indices: &[A],
    shape: &[usize],
    strides: &[usize],
) -> AtlasNdResult<usize> {
    debug_assert_eq!(shape.len(), strides.len());
    if indices.len() != shape.len() {
        return Err(AtlasNdError::DimensionMismatch {
            expected: shape.len(),
            actual: indices.len(),
        });
    }

    let mut offset = base_offset;

    for (axis, ((index, dim), stride)) in
        indices.iter().zip(shape.iter()).zip(strides.iter()).enumerate()
    {
        let normalized = normalize_scalar_index(*index, axis, *dim)?;
        let axis_offset = normalized.checked_mul(*stride).ok_or_else(|| {
            AtlasNdError::ShapeOverflow { op: "index offset", shape: shape.to_vec() }
        })?;
        offset = offset.checked_add(axis_offset).ok_or_else(|| AtlasNdError::ShapeOverflow {
            op: "index offset",
            shape: shape.to_vec(),
        })?;
    }

    Ok(offset)
}

pub(crate) fn normalize_insertion_axis<A: AxisIndex>(axis: A, ndim: usize) -> AtlasNdResult<usize> {
    let axis = axis.try_into_i64().ok_or(AtlasNdError::InvalidAxis { axis: i64::MAX, ndim })?;
    let upper_bound = ndim as i64;
    let normalized = if axis < 0 { upper_bound + 1 + axis } else { axis };

    if normalized < 0 || normalized > upper_bound {
        return Err(AtlasNdError::InvalidAxis { axis, ndim });
    }

    Ok(normalized as usize)
}

#[cfg(test)]
mod tests {
    use crate::AtlasNdError;

    use super::{
        normalize_and_offset_indices, normalize_axis, normalize_insertion_axis,
        normalize_scalar_index,
    };

    #[test]
    fn normalize_axis_supports_positive_and_negative_indices() {
        assert_eq!(normalize_axis(0_i32, 3).unwrap(), 0);
        assert_eq!(normalize_axis(2_i32, 3).unwrap(), 2);
        assert_eq!(normalize_axis(-1_i32, 3).unwrap(), 2);
        assert_eq!(normalize_axis(-3_i32, 3).unwrap(), 0);
    }

    #[test]
    fn normalize_axis_rejects_out_of_bounds_indices() {
        assert_eq!(
            normalize_axis(3_i32, 3).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: 3, ndim: 3 }
        );
        assert_eq!(
            normalize_axis(-4_i32, 3).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: -4, ndim: 3 }
        );
    }

    #[test]
    fn normalize_axis_rejects_unsigned_values_above_i64_max() {
        assert_eq!(
            normalize_axis(usize::MAX, 3).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: i64::MAX, ndim: 3 }
        );
        assert_eq!(
            normalize_axis(u64::MAX, 3).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: i64::MAX, ndim: 3 }
        );
    }

    #[test]
    fn normalize_insertion_axis_supports_positive_and_negative_indices() {
        assert_eq!(normalize_insertion_axis(0_i32, 2).unwrap(), 0);
        assert_eq!(normalize_insertion_axis(1_i32, 2).unwrap(), 1);
        assert_eq!(normalize_insertion_axis(2_i32, 2).unwrap(), 2);
        assert_eq!(normalize_insertion_axis(-1_i32, 2).unwrap(), 2);
        assert_eq!(normalize_insertion_axis(-2_i32, 2).unwrap(), 1);
        assert_eq!(normalize_insertion_axis(-3_i32, 2).unwrap(), 0);
    }

    #[test]
    fn normalize_insertion_axis_rejects_out_of_bounds_indices() {
        assert_eq!(
            normalize_insertion_axis(3_i32, 2).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: 3, ndim: 2 }
        );
        assert_eq!(
            normalize_insertion_axis(-4_i32, 2).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: -4, ndim: 2 }
        );
    }

    #[test]
    fn normalize_scalar_index_supports_positive_and_negative_indices() {
        assert_eq!(normalize_scalar_index(0_i32, 0, 3).unwrap(), 0);
        assert_eq!(normalize_scalar_index(2_i32, 0, 3).unwrap(), 2);
        assert_eq!(normalize_scalar_index(-1_i32, 0, 3).unwrap(), 2);
        assert_eq!(normalize_scalar_index(-3_i32, 0, 3).unwrap(), 0);
    }

    #[test]
    fn normalize_scalar_index_rejects_out_of_bounds_indices() {
        assert_eq!(
            normalize_scalar_index(3_i32, 0, 3).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 0, index: 3, dim: 3 }
        );
        assert_eq!(
            normalize_scalar_index(-4_i32, 0, 3).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 0, index: -4, dim: 3 }
        );
    }

    #[test]
    fn normalize_scalar_index_rejects_unsigned_values_above_i64_max() {
        assert_eq!(
            normalize_scalar_index(usize::MAX, 1, 3).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 1, index: i64::MAX, dim: 3 }
        );
    }

    #[test]
    fn normalize_and_offset_indices_centralizes_dimension_and_bounds_checks() {
        assert_eq!(normalize_and_offset_indices(5, &[1_i32, -1], &[2, 3], &[3, 1]).unwrap(), 10);
        assert_eq!(
            normalize_and_offset_indices(0, &[0_i32], &[2, 3], &[3, 1]).unwrap_err(),
            AtlasNdError::DimensionMismatch { expected: 2, actual: 1 }
        );
        assert_eq!(
            normalize_and_offset_indices(0, &[0_i32, 3], &[2, 3], &[3, 1]).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 1, index: 3, dim: 3 }
        );
    }
}
