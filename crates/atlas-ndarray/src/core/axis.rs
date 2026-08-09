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

#[cfg(test)]
mod tests {
    use crate::AtlasNdError;

    use super::normalize_axis;

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
}
