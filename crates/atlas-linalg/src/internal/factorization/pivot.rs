use num_traits::Float;

pub(crate) fn find_pivot_row<T: Float>(matrix: &[T], n: usize, pivot_col: usize) -> usize {
    let mut pivot_row = pivot_col;
    let mut pivot_value = matrix[pivot_col * n + pivot_col].abs();

    for candidate in (pivot_col + 1)..n {
        let value = matrix[candidate * n + pivot_col].abs();

        if value > pivot_value {
            pivot_value = value;
            pivot_row = candidate;
        }
    }

    pivot_row
}

pub(crate) fn swap_rows<T>(matrix: &mut [T], cols: usize, left: usize, right: usize) {
    if left == right {
        return;
    }

    let left_start = left * cols;
    let right_start = right * cols;
    let (head, tail) = matrix.split_at_mut(right_start);
    let left_row = &mut head[left_start..left_start + cols];
    let right_row = &mut tail[..cols];

    left_row.swap_with_slice(right_row);
}

pub(crate) fn swap_l_prefix_rows<T>(
    matrix: &mut [T],
    cols: usize,
    left: usize,
    right: usize,
    end: usize,
) {
    if left == right || end == 0 {
        return;
    }

    let left_start = left * cols;
    let right_start = right * cols;
    let (head, tail) = matrix.split_at_mut(right_start);
    let left_prefix = &mut head[left_start..left_start + end];
    let right_prefix = &mut tail[..end];

    left_prefix.swap_with_slice(right_prefix);
}
