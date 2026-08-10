use crate::{AtlasNdError, AtlasNdResult, Numeric, ShapeArg, view::ArrayView};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceRange {
    pub start: Option<i64>,
    pub end: Option<i64>,
    pub step: i64,
}

impl SliceRange {
    pub const fn new(start: Option<i64>, end: Option<i64>, step: i64) -> Self {
        Self { start, end, step }
    }

    pub const fn between(start: Option<i64>, end: Option<i64>) -> Self {
        Self { start, end, step: 1 }
    }

    pub const fn full() -> Self {
        Self { start: None, end: None, step: 1 }
    }
}

impl Default for SliceRange {
    fn default() -> Self {
        Self::full()
    }
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn slice<I, S>(&self, starts: I, new_shape: S) -> AtlasNdResult<ArrayView<'a, T>>
    where
        I: AsRef<[usize]>,
        S: ShapeArg,
    {
        let starts = starts.as_ref();
        let new_shape = new_shape.into_shape_vec();
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

            if *len == 0 {
                break;
            }

            let axis_offset = start.checked_mul(*stride).ok_or_else(|| {
                AtlasNdError::ShapeOverflow { op: "slice offset", shape: self.shape.clone() }
            })?;
            offset = offset.checked_add(axis_offset).ok_or_else(|| {
                AtlasNdError::ShapeOverflow { op: "slice offset", shape: self.shape.clone() }
            })?;
        }

        ArrayView::from_parts(self.data, offset, new_shape, self.strides.clone())
    }

    pub fn slice_ranges<R>(&self, ranges: R) -> AtlasNdResult<ArrayView<'a, T>>
    where
        R: AsRef<[SliceRange]>,
    {
        let ranges = ranges.as_ref();
        debug_assert_eq!(self.shape.len(), self.strides.len());
        if ranges.len() != self.shape.len() {
            return Err(AtlasNdError::DimensionMismatch {
                expected: self.shape.len(),
                actual: ranges.len(),
            });
        }

        let mut offset = self.offset;
        let mut new_shape = Vec::with_capacity(self.shape.len());
        let mut new_strides = Vec::with_capacity(self.strides.len());

        for (axis, ((range, dim), stride)) in
            ranges.iter().zip(self.shape.iter()).zip(self.strides.iter()).enumerate()
        {
            let normalized = normalize_slice_range(*range, axis, *dim)?;
            let axis_offset = normalized.start.checked_mul(*stride).ok_or_else(|| {
                AtlasNdError::ShapeOverflow { op: "slice offset", shape: self.shape.clone() }
            })?;
            let stepped_stride = stride.checked_mul(normalized.step).ok_or_else(|| {
                AtlasNdError::ShapeOverflow { op: "slice stride", shape: self.shape.clone() }
            })?;
            offset = offset.checked_add(axis_offset).ok_or_else(|| {
                AtlasNdError::ShapeOverflow { op: "slice offset", shape: self.shape.clone() }
            })?;
            new_shape.push(normalized.len);
            new_strides.push(stepped_stride);
        }

        ArrayView::from_parts(self.data, offset, new_shape, new_strides)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct NormalizedSliceRange {
    start: usize,
    len: usize,
    step: usize,
}

fn normalize_slice_range(
    range: SliceRange,
    _axis: usize,
    dim: usize,
) -> AtlasNdResult<NormalizedSliceRange> {
    if range.step == 0 {
        return Err(AtlasNdError::InvalidArgument { op: "slice", reason: "step must be non-zero" });
    }

    if range.step < 0 {
        return Err(AtlasNdError::InvalidArgument {
            op: "slice",
            reason: "negative step is not supported",
        });
    }

    let step = usize::try_from(range.step).expect("positive i64 step always fits into usize");
    let dim_i128 = dim as i128;
    let start = normalize_slice_bound(range.start, 0, dim_i128);
    let end = normalize_slice_bound(range.end, dim_i128, dim_i128);
    let len = if start >= end {
        0
    } else {
        let span = end - start;
        usize::try_from(((span - 1) / step as i128) + 1)
            .expect("normalized positive slice length always fits into usize")
    };

    Ok(NormalizedSliceRange {
        start: usize::try_from(start).expect("normalized slice start always fits into usize"),
        len,
        step,
    })
}

fn normalize_slice_bound(bound: Option<i64>, default: i128, dim: i128) -> i128 {
    let bound = bound.map_or(default, i128::from);
    let normalized = if bound < 0 { dim + bound } else { bound };
    normalized.clamp(0, dim)
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray, view::slicing::SliceRange};

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
        let array = NDArray::new(vec![2, 3], 0_i32).unwrap();
        let view = array.view();
        let error = view.slice([0, 2], [2, 2]).unwrap_err();

        assert_eq!(error, AtlasNdError::InvalidSlice { axis: 1, start: 2, len: 2, dim: 3 });
    }

    #[test]
    fn slice_rejects_rank_mismatch() {
        let array = NDArray::new(vec![2, 3], 0_i32).unwrap();
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
        assert_eq!(slice.offset(), 0);
        assert_eq!(slice.len(), 0);
        assert!(slice.is_empty());
    }

    #[test]
    fn slice_keeps_prefix_offset_for_zero_length_suffix_axes() {
        let array = NDArray::from_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view();
        let slice = view.slice([1, 3], [1, 0]).unwrap();

        assert_eq!(slice.shape(), &[1, 0]);
        assert_eq!(slice.offset(), 3);
        assert!(slice.is_empty());
    }

    #[test]
    fn slice_reports_new_shape_rank_mismatch_consistently() {
        let array = NDArray::new(vec![2, 3], 0_i32).unwrap();
        let view = array.view();

        assert_eq!(
            view.slice([0, 0], [1]).unwrap_err(),
            AtlasNdError::DimensionMismatch { expected: 2, actual: 1 }
        );
    }

    #[test]
    fn slice_ranges_builds_step_aware_views_with_normalized_bounds() {
        let array = NDArray::from_vec([2, 5], vec![0_i32, 1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let view = array.view();
        let slice = view
            .slice_ranges([
                SliceRange::between(Some(0), Some(2)),
                SliceRange::new(Some(1), None, 2),
            ])
            .unwrap();

        assert_eq!(slice.shape(), &[2, 2]);
        assert_eq!(slice.strides(), &[5, 2]);
        assert_eq!(slice.offset(), 1);
        assert_eq!(*slice.get(&[0, 0]).unwrap(), 1);
        assert_eq!(*slice.get(&[0, 1]).unwrap(), 3);
        assert_eq!(*slice.get(&[1, 1]).unwrap(), 8);
    }

    #[test]
    fn slice_ranges_supports_negative_bounds_and_clamps_out_of_range_values() {
        let array = NDArray::from_vec([6], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let slice = array.view().slice_ranges([SliceRange::new(Some(-5), Some(99), 2)]).unwrap();

        assert_eq!(slice.shape(), &[3]);
        assert_eq!(slice.strides(), &[2]);
        assert_eq!(slice.offset(), 1);
        assert_eq!(*slice.get(&[0]).unwrap(), 1);
        assert_eq!(*slice.get(&[1]).unwrap(), 3);
        assert_eq!(*slice.get(&[2]).unwrap(), 5);
    }

    #[test]
    fn slice_ranges_supports_zero_length_and_full_axis_defaults() {
        let array = NDArray::from_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let full = array.view().slice_ranges([SliceRange::full(), SliceRange::full()]).unwrap();
        let empty = array
            .view()
            .slice_ranges([SliceRange::new(Some(5), Some(1), 1), SliceRange::full()])
            .unwrap();

        assert_eq!(full.shape(), &[2, 3]);
        assert_eq!(full.strides(), &[3, 1]);
        assert_eq!(full.offset(), 0);

        assert_eq!(empty.shape(), &[0, 3]);
        assert_eq!(empty.strides(), &[3, 1]);
        assert_eq!(empty.offset(), 6);
        assert!(empty.is_empty());
    }

    #[test]
    fn slice_ranges_rejects_invalid_step_and_rank_mismatch() {
        let array = NDArray::from_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view();

        assert_eq!(
            view.slice_ranges([SliceRange::new(None, None, 0), SliceRange::full()]).unwrap_err(),
            AtlasNdError::InvalidArgument { op: "slice", reason: "step must be non-zero" }
        );
        assert_eq!(
            view.slice_ranges([SliceRange::new(None, None, -1), SliceRange::full()]).unwrap_err(),
            AtlasNdError::InvalidArgument { op: "slice", reason: "negative step is not supported" }
        );
        assert_eq!(
            view.slice_ranges([SliceRange::full()]).unwrap_err(),
            AtlasNdError::DimensionMismatch { expected: 2, actual: 1 }
        );
    }
}
