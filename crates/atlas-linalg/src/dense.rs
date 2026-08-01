use atlas_ndarray::{NDArray, Numeric};

use crate::{
    error::{AtlasLinalgError, AtlasLinalgResult},
    operand::LinalgOperand,
};

pub fn dot<'a, T, L, R>(lhs: L, rhs: R) -> AtlasLinalgResult<T>
where
    T: Numeric + 'a,
    L: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'a, T>>,
{
    let lhs = lhs.into();
    let rhs = rhs.into();

    match (lhs.shape(), rhs.shape()) {
        ([lhs_len], [rhs_len]) if lhs_len == rhs_len => {
            let lhs = vector_ref(&lhs);
            let rhs = vector_ref(&rhs);

            Ok(dot_kernel(lhs, rhs))
        }
        ([..], [..]) if lhs.ndim() == 1 && rhs.ndim() == 1 => {
            Err(AtlasLinalgError::ShapeMismatch {
                op: "dot",
                left: lhs.shape().to_vec(),
                right: rhs.shape().to_vec(),
                reason: "vector lengths must match",
            })
        }
        _ => Err(AtlasLinalgError::InvalidOperandRank {
            op: "dot",
            left: lhs.ndim(),
            right: rhs.ndim(),
        }),
    }
}

pub fn matmul<'a, T, L, R>(lhs: L, rhs: R) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + 'a,
    L: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'a, T>>,
{
    let lhs = lhs.into();
    let rhs = rhs.into();

    match (lhs.ndim(), rhs.ndim()) {
        (1, 1) => matmul_vector_vector(&lhs, &rhs),
        (1, 2) => matmul_vector_matrix(&lhs, &rhs),
        (2, 1) => matmul_matrix_vector(&lhs, &rhs),
        (2, 2) => matmul_matrix_matrix(&lhs, &rhs),
        _ => Err(AtlasLinalgError::InvalidOperandRank {
            op: "matmul",
            left: lhs.ndim(),
            right: rhs.ndim(),
        }),
    }
}

fn matmul_vector_vector<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let lhs = vector_ref(lhs);
    let rhs = vector_ref(rhs);

    if lhs.len != rhs.len {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![lhs.len],
            right: vec![rhs.len],
            reason: "vector lengths must match",
        });
    }

    let value = dot_kernel(lhs, rhs);
    wrap_scalar(value)
}

fn matmul_vector_matrix<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let lhs = vector_ref(lhs);
    let rhs = matrix_ref(rhs);

    if lhs.len != rhs.rows {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![lhs.len],
            right: vec![rhs.rows, rhs.cols],
            reason: "vector length must match matrix row count",
        });
    }

    let data = if lhs.is_contiguous() && rhs.is_row_major_contiguous() {
        matmul_vector_matrix_row_major(lhs, rhs)
    } else if lhs.is_contiguous() && rhs.is_col_major_contiguous() {
        matmul_vector_matrix_col_major(lhs, rhs)
    } else {
        matmul_vector_matrix_generic(lhs, rhs)
    };

    Ok(NDArray::from_shape_vec([rhs.cols], data)?)
}

fn matmul_matrix_vector<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let lhs = matrix_ref(lhs);
    let rhs = vector_ref(rhs);

    if lhs.cols != rhs.len {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![lhs.rows, lhs.cols],
            right: vec![rhs.len],
            reason: "matrix column count must match vector length",
        });
    }

    let data = if lhs.is_row_major_contiguous() && rhs.is_contiguous() {
        matmul_matrix_vector_row_major(lhs, rhs)
    } else if lhs.is_col_major_contiguous() && rhs.is_contiguous() {
        matmul_matrix_vector_col_major(lhs, rhs)
    } else {
        matmul_matrix_vector_generic(lhs, rhs)
    };

    Ok(NDArray::from_shape_vec([lhs.rows], data)?)
}

fn matmul_matrix_matrix<T: Numeric>(
    lhs: &LinalgOperand<'_, T>,
    rhs: &LinalgOperand<'_, T>,
) -> AtlasLinalgResult<NDArray<T>> {
    let lhs = matrix_ref(lhs);
    let rhs = matrix_ref(rhs);

    if lhs.cols != rhs.rows {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "matmul",
            left: vec![lhs.rows, lhs.cols],
            right: vec![rhs.rows, rhs.cols],
            reason: "left matrix column count must match right matrix row count",
        });
    }

    let data = if lhs.is_row_major_contiguous() && rhs.is_row_major_contiguous() {
        matmul_matrix_matrix_row_major(lhs, rhs)
    } else if lhs.is_col_major_contiguous() && rhs.is_row_major_contiguous() {
        matmul_matrix_matrix_lhs_col_major(lhs, rhs)
    } else if lhs.is_row_major_contiguous() && rhs.is_col_major_contiguous() {
        matmul_matrix_matrix_rhs_col_major(lhs, rhs)
    } else {
        matmul_matrix_matrix_generic(lhs, rhs)
    };

    Ok(NDArray::from_shape_vec([lhs.rows, rhs.cols], data)?)
}

fn wrap_scalar<T: Numeric>(value: T) -> AtlasLinalgResult<NDArray<T>> {
    Ok(NDArray::from_shape_vec([], vec![value])?)
}

#[derive(Clone, Copy)]
struct VectorRef<'a, T: Numeric> {
    data: &'a [T],
    offset: usize,
    len: usize,
    stride: usize,
}

impl<'a, T: Numeric> VectorRef<'a, T> {
    fn is_contiguous(&self) -> bool {
        self.stride == 1
    }

    fn value_at(&self, index: usize) -> T {
        self.data[self.offset + index * self.stride]
    }

    fn contiguous_slice(&self) -> &'a [T] {
        debug_assert!(self.is_contiguous());
        &self.data[self.offset..self.offset + self.len]
    }
}

#[derive(Clone, Copy)]
struct MatrixRef<'a, T: Numeric> {
    data: &'a [T],
    offset: usize,
    rows: usize,
    cols: usize,
    row_stride: usize,
    col_stride: usize,
}

impl<'a, T: Numeric> MatrixRef<'a, T> {
    fn is_row_major_contiguous(&self) -> bool {
        self.col_stride == 1 && self.row_stride == self.cols
    }

    fn is_col_major_contiguous(&self) -> bool {
        self.row_stride == 1 && self.col_stride == self.rows
    }

    fn value_at(&self, row: usize, col: usize) -> T {
        self.data[self.offset + row * self.row_stride + col * self.col_stride]
    }

    fn contiguous_row_slice(&self, row: usize) -> &'a [T] {
        debug_assert!(self.is_row_major_contiguous());
        let start = self.offset + row * self.row_stride;

        &self.data[start..start + self.cols]
    }

    fn contiguous_col_slice(&self, col: usize) -> &'a [T] {
        debug_assert!(self.is_col_major_contiguous());
        let start = self.offset + col * self.col_stride;

        &self.data[start..start + self.rows]
    }
}

fn vector_ref<'a, T: Numeric>(operand: &'a LinalgOperand<'a, T>) -> VectorRef<'a, T> {
    VectorRef {
        data: operand.data(),
        offset: operand.offset(),
        len: operand.shape()[0],
        stride: operand.strides()[0],
    }
}

fn matrix_ref<'a, T: Numeric>(operand: &'a LinalgOperand<'a, T>) -> MatrixRef<'a, T> {
    MatrixRef {
        data: operand.data(),
        offset: operand.offset(),
        rows: operand.shape()[0],
        cols: operand.shape()[1],
        row_stride: operand.strides()[0],
        col_stride: operand.strides()[1],
    }
}

fn dot_kernel<T: Numeric>(lhs: VectorRef<'_, T>, rhs: VectorRef<'_, T>) -> T {
    if lhs.is_contiguous() && rhs.is_contiguous() {
        dot_contiguous(lhs.contiguous_slice(), rhs.contiguous_slice())
    } else {
        dot_strided(lhs, rhs)
    }
}

fn dot_contiguous<T: Numeric>(lhs: &[T], rhs: &[T]) -> T {
    let mut total = T::zero();

    for index in 0..lhs.len() {
        total += lhs[index] * rhs[index];
    }

    total
}

fn dot_strided<T: Numeric>(lhs: VectorRef<'_, T>, rhs: VectorRef<'_, T>) -> T {
    let mut total = T::zero();

    for index in 0..lhs.len {
        total += lhs.value_at(index) * rhs.value_at(index);
    }

    total
}

fn matmul_vector_matrix_row_major<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); rhs.cols];

    for k in 0..lhs.len {
        let lhs_value = lhs.value_at(k);
        let rhs_row = rhs.contiguous_row_slice(k);

        for col in 0..rhs.cols {
            data[col] += lhs_value * rhs_row[col];
        }
    }

    data
}

fn matmul_vector_matrix_col_major<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let lhs = lhs.contiguous_slice();
    let mut data = vec![T::zero(); rhs.cols];

    for (col, output) in data.iter_mut().enumerate() {
        *output = dot_contiguous(lhs, rhs.contiguous_col_slice(col));
    }

    data
}

fn matmul_vector_matrix_generic<T: Numeric>(
    lhs: VectorRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); rhs.cols];

    for (col, output) in data.iter_mut().enumerate() {
        let mut total = T::zero();

        for k in 0..lhs.len {
            total += lhs.value_at(k) * rhs.value_at(k, col);
        }

        *output = total;
    }

    data
}

fn matmul_matrix_vector_row_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
) -> Vec<T> {
    let rhs = rhs.contiguous_slice();
    let mut data = vec![T::zero(); lhs.rows];

    for (row, output) in data.iter_mut().enumerate() {
        *output = dot_contiguous(lhs.contiguous_row_slice(row), rhs);
    }

    data
}

fn matmul_matrix_vector_col_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows];

    for k in 0..lhs.cols {
        let rhs_value = rhs.value_at(k);
        let lhs_col = lhs.contiguous_col_slice(k);

        for row in 0..lhs.rows {
            data[row] += lhs_col[row] * rhs_value;
        }
    }

    data
}

fn matmul_matrix_vector_generic<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: VectorRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows];

    for (row, output) in data.iter_mut().enumerate() {
        let mut total = T::zero();

        for k in 0..lhs.cols {
            total += lhs.value_at(row, k) * rhs.value_at(k);
        }

        *output = total;
    }

    data
}

fn matmul_matrix_matrix_row_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows * rhs.cols];

    for row in 0..lhs.rows {
        let lhs_row = lhs.contiguous_row_slice(row);
        let out_row = &mut data[row * rhs.cols..(row + 1) * rhs.cols];

        for (k, lhs_value) in lhs_row.iter().copied().enumerate() {
            let rhs_row = rhs.contiguous_row_slice(k);

            for col in 0..rhs.cols {
                out_row[col] += lhs_value * rhs_row[col];
            }
        }
    }

    data
}

fn matmul_matrix_matrix_lhs_col_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows * rhs.cols];

    for k in 0..lhs.cols {
        let lhs_col = lhs.contiguous_col_slice(k);
        let rhs_row = rhs.contiguous_row_slice(k);

        for row in 0..lhs.rows {
            let lhs_value = lhs_col[row];
            let out_row = &mut data[row * rhs.cols..(row + 1) * rhs.cols];

            for col in 0..rhs.cols {
                out_row[col] += lhs_value * rhs_row[col];
            }
        }
    }

    data
}

fn matmul_matrix_matrix_rhs_col_major<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows * rhs.cols];

    for row in 0..lhs.rows {
        let lhs_row = lhs.contiguous_row_slice(row);
        let out_row = &mut data[row * rhs.cols..(row + 1) * rhs.cols];

        for (col, output) in out_row.iter_mut().enumerate() {
            *output = dot_contiguous(lhs_row, rhs.contiguous_col_slice(col));
        }
    }

    data
}

fn matmul_matrix_matrix_generic<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows * rhs.cols];

    for row in 0..lhs.rows {
        let out_row = &mut data[row * rhs.cols..(row + 1) * rhs.cols];

        for (col, output) in out_row.iter_mut().enumerate() {
            let mut total = T::zero();

            for k in 0..lhs.cols {
                total += lhs.value_at(row, k) * rhs.value_at(k, col);
            }

            *output = total;
        }
    }

    data
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{dot, error::AtlasLinalgError, matmul};

    #[test]
    fn dot_rejects_non_vector_inputs_and_mismatched_lengths() {
        let lhs = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let rhs = NDArray::from_shape_vec([2], vec![4_i32, 5]).unwrap();
        let matrix = NDArray::from_shape_vec([1, 3], vec![1_i32, 2, 3]).unwrap();

        assert_eq!(
            dot(&lhs, &rhs).unwrap_err(),
            AtlasLinalgError::ShapeMismatch {
                op: "dot",
                left: vec![3],
                right: vec![2],
                reason: "vector lengths must match",
            }
        );
        assert_eq!(
            dot(&lhs, &matrix).unwrap_err(),
            AtlasLinalgError::InvalidOperandRank { op: "dot", left: 1, right: 2 }
        );
    }

    #[test]
    fn matmul_supports_vector_and_matrix_operands() {
        let lhs_vec = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let rhs_vec = NDArray::from_shape_vec([3], vec![4_i32, 5, 6]).unwrap();
        let matrix = NDArray::from_shape_vec([3, 2], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let left_matrix = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let right_matrix = NDArray::from_shape_vec([3, 2], vec![7_i32, 8, 9, 10, 11, 12]).unwrap();

        assert_eq!(dot(&lhs_vec, &rhs_vec).unwrap(), 32);
        assert_eq!(matmul(&lhs_vec, &rhs_vec).unwrap().shape(), &[] as &[usize]);
        assert_eq!(matmul(&lhs_vec, &rhs_vec).unwrap().data(), &[32]);
        assert_eq!(matmul(&lhs_vec, &matrix).unwrap().data(), &[22, 28]);
        assert_eq!(matmul(&left_matrix, &rhs_vec).unwrap().data(), &[32, 77]);
        assert_eq!(matmul(&left_matrix, &right_matrix).unwrap().shape(), &[2, 2]);
        assert_eq!(matmul(&left_matrix, &right_matrix).unwrap().data(), &[58, 64, 139, 154]);
    }

    #[test]
    fn matmul_vector_vector_reports_matmul_shape_mismatch() {
        let lhs = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let rhs = NDArray::from_shape_vec([2], vec![4_i32, 5]).unwrap();

        assert_eq!(
            matmul(&lhs, &rhs).unwrap_err(),
            AtlasLinalgError::ShapeMismatch {
                op: "matmul",
                left: vec![3],
                right: vec![2],
                reason: "vector lengths must match",
            }
        );
    }

    #[test]
    fn matmul_supports_transposed_views_and_strided_fallbacks() {
        let left_base = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let rhs = NDArray::from_shape_vec([2, 2], vec![7_i32, 8, 9, 10]).unwrap();
        let rhs_base = NDArray::from_shape_vec([2, 3], vec![7_i32, 9, 11, 8, 10, 12]).unwrap();
        let generic_base = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let vector = NDArray::from_shape_vec([2], vec![10_i32, 20]).unwrap();

        let left_transposed = left_base.view().transpose();
        let right_transposed = rhs_base.view().transpose();
        let sliced = generic_base.view().slice([0, 1], [2, 2]).unwrap();

        assert_eq!(matmul(left_transposed, &rhs).unwrap().data(), &[43, 48, 59, 66, 75, 84]);
        assert_eq!(matmul(&left_base, right_transposed).unwrap().data(), &[58, 64, 139, 154]);
        assert_eq!(matmul(sliced, &vector).unwrap().data(), &[50, 140]);
    }

    #[test]
    fn dot_and_matmul_match_for_owned_and_view_vector_operands() {
        let lhs = NDArray::from_shape_vec([4], vec![1_i32, 2, 3, 4]).unwrap();
        let rhs = NDArray::from_shape_vec([4], vec![5_i32, 6, 7, 8]).unwrap();
        let lhs_view = lhs.view().slice([0], [4]).unwrap();
        let rhs_view = rhs.view().slice([0], [4]).unwrap();

        assert_eq!(dot(&lhs, &rhs).unwrap(), dot(lhs_view.clone(), rhs_view.clone()).unwrap());

        let owned = matmul(&lhs, &rhs).unwrap();
        let viewed = matmul(lhs_view, rhs_view).unwrap();

        assert_eq!(owned.shape(), &[] as &[usize]);
        assert_eq!(owned.shape(), viewed.shape());
        assert_eq!(owned.data(), viewed.data());
    }
}
