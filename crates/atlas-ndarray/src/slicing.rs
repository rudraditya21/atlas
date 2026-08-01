use super::{
    error::{AtlasNdError, AtlasNdResult},
    traits::Numeric,
    view::ArrayView,
};

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn slice<I, S>(&self, starts: I, new_shape: S) -> AtlasNdResult<ArrayView<'a, T>>
    where
        I: AsRef<[usize]>,
        S: AsRef<[usize]>,
    {
        let starts = starts.as_ref();
        let new_shape = new_shape.as_ref().to_vec();
        debug_assert_eq!(self.shape.len(), self.strides.len());
        if starts.len() != self.shape.len() {
            return Err(AtlasNdError::DimensionMismatch {
                expected: self.shape.len(),
                actual: starts.len(),
            });
        }

        if new_shape.len() != self.shape.len() {
            return Err(AtlasNdError::DimensionMismatch {
                expected: self.shape.len(),
                actual: new_shape.len(),
            });
        }

        let mut offset = self.offset;

        for (axis, (((start, len), dim), stride)) in starts
            .iter()
            .zip(new_shape.iter())
            .zip(self.shape.iter())
            .zip(self.strides.iter())
            .enumerate()
        {
            let end = start.checked_add(*len).ok_or(AtlasNdError::InvalidSlice {
                axis,
                start: *start,
                len: *len,
                dim: *dim,
            })?;

            if *start > *dim || end > *dim {
                return Err(AtlasNdError::InvalidSlice {
                    axis,
                    start: *start,
                    len: *len,
                    dim: *dim,
                });
            }

            offset += start * stride;
        }

        Ok(ArrayView { data: self.data, offset, shape: new_shape, strides: self.strides.clone() })
    }
}

#[cfg(test)]
mod tests {
    use crate::{array::NDArray, error::AtlasNdError};

    #[test]
    fn slice_builds_a_view_with_checked_bounds() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view();
        let slice = view.slice([0, 1], [2, 2]).unwrap();

        assert_eq!(slice.shape(), &[2, 2]);
        assert_eq!(slice.strides(), &[3, 1]);
        assert!(!slice.is_contiguous());
        assert_eq!(*slice.get(&[0, 0]).unwrap(), 1);
        assert_eq!(*slice.get(&[1, 1]).unwrap(), 5);
    }

    #[test]
    fn slice_rejects_invalid_extent() {
        let array = NDArray::new(vec![2, 3], 0_i32);
        let view = array.view();
        let error = view.slice([0, 2], [2, 2]).unwrap_err();

        assert_eq!(error, AtlasNdError::InvalidSlice { axis: 1, start: 2, len: 2, dim: 3 });
    }

    #[test]
    fn slice_rejects_rank_mismatch() {
        let array = NDArray::new(vec![2, 3], 0_i32);
        let view = array.view();
        let error = view.slice([0], [1, 1]).unwrap_err();

        assert_eq!(error, AtlasNdError::DimensionMismatch { expected: 2, actual: 1 });
    }

    #[test]
    fn slice_accepts_zero_length_ranges_at_axis_boundaries() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view();
        let slice = view.slice([2, 3], [0, 0]).unwrap();

        assert_eq!(slice.shape(), &[0, 0]);
        assert_eq!(slice.strides(), &[3, 1]);
        assert_eq!(slice.len(), 0);
        assert!(slice.is_empty());
    }

    #[test]
    fn slice_reports_new_shape_rank_mismatch_consistently() {
        let array = NDArray::new(vec![2, 3], 0_i32);
        let view = array.view();

        assert_eq!(
            view.slice([0, 0], [1]).unwrap_err(),
            AtlasNdError::DimensionMismatch { expected: 2, actual: 1 }
        );
    }
}
