pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
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
    use super::{compute_strides, element_count};

    #[test]
    fn element_count_handles_scalar_and_zero_sized_shapes() {
        assert_eq!(element_count(&[]), 1);
        assert_eq!(element_count(&[5]), 5);
        assert_eq!(element_count(&[2, 0, 4]), 0);
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
