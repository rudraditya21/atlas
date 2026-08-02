use crate::{AtlasNdError, AtlasNdResult};

pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub fn checked_element_count(shape: &[usize]) -> AtlasNdResult<usize> {
    shape.iter().try_fold(1usize, |count, &dim| {
        count.checked_mul(dim).ok_or_else(|| AtlasNdError::ShapeOverflow {
            op: "element count",
            shape: shape.to_vec(),
        })
    })
}

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    // Row-major contiguous layout: the last axis is unit-stride and each axis
    // to the left spans the full extent of the axis immediately to its right.
    let mut strides = vec![1; shape.len()];

    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    strides
}

#[cfg(test)]
mod tests {
    use crate::AtlasNdError;

    use super::{checked_element_count, compute_strides, element_count};

    #[test]
    fn element_count_handles_scalar_and_zero_sized_shapes() {
        assert_eq!(element_count(&[]), 1);
        assert_eq!(element_count(&[5]), 5);
        assert_eq!(element_count(&[2, 0, 4]), 0);
    }

    #[test]
    fn checked_element_count_handles_scalar_and_zero_sized_shapes() {
        assert_eq!(checked_element_count(&[]).unwrap(), 1);
        assert_eq!(checked_element_count(&[5]).unwrap(), 5);
        assert_eq!(checked_element_count(&[2, 0, 4]).unwrap(), 0);
    }

    #[test]
    fn checked_element_count_reports_overflow_explicitly() {
        assert_eq!(
            checked_element_count(&[usize::MAX, 2]).unwrap_err(),
            AtlasNdError::ShapeOverflow { op: "element count", shape: vec![usize::MAX, 2] }
        );
    }

    #[test]
    fn compute_strides_handles_scalar_shape() {
        assert_eq!(compute_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn compute_strides_handles_one_dimensional_shape() {
        assert_eq!(compute_strides(&[5]), vec![1]);
    }

    #[test]
    fn compute_strides_handles_two_dimensional_shape() {
        assert_eq!(compute_strides(&[2, 3]), vec![3, 1]);
    }

    #[test]
    fn compute_strides_handles_three_dimensional_shape() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }
}
