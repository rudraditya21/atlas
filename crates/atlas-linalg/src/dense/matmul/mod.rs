mod col_major;
mod dispatch;
mod generic;
mod matrix_matrix;
mod matrix_vector;
mod row_major;
mod vector_matrix;
mod vector_vector;

use atlas_ndarray::{NDArray, Numeric};

use crate::core::{AtlasLinalgResult, LinalgOperand};

pub fn matmul<'a, T, L, R>(lhs: L, rhs: R) -> AtlasLinalgResult<NDArray<T>>
where
    T: Numeric + 'a,
    L: Into<LinalgOperand<'a, T>>,
    R: Into<LinalgOperand<'a, T>>,
{
    let lhs = lhs.into();
    let rhs = rhs.into();

    dispatch::dispatch_matmul(&lhs, &rhs)
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::{
        dispatch::should_parallelize_matmul_for_threads,
        matmul,
        matrix_matrix::{
            matmul_matrix_matrix_generic, matmul_matrix_matrix_lhs_col_major,
            matmul_matrix_matrix_rhs_col_major, matmul_matrix_matrix_row_major,
        },
        matrix_vector::{
            matmul_matrix_vector_col_major, matmul_matrix_vector_generic,
            matmul_matrix_vector_row_major,
        },
        row_major::should_use_blocked_row_major_matmul_for_dims,
        vector_matrix::{
            matmul_vector_matrix_col_major, matmul_vector_matrix_generic,
            matmul_vector_matrix_row_major,
        },
    };
    use crate::internal::dense::{matrix_ref, vector_ref};
    use crate::{AtlasLinalgError, LinalgOperand};

    fn vector_row_major(values: &[i32]) -> NDArray<i32> {
        NDArray::from_shape_vec([values.len()], values.to_vec()).unwrap()
    }

    fn matrix_row_major(rows: usize, cols: usize, values: &[i32]) -> NDArray<i32> {
        NDArray::from_shape_vec([rows, cols], values.to_vec()).unwrap()
    }

    fn matrix_col_major_from_rows(rows: usize, cols: usize, values: &[i32]) -> NDArray<i32> {
        let mut data = Vec::with_capacity(rows * cols);

        for col in 0..cols {
            for row in 0..rows {
                data.push(values[row * cols + col]);
            }
        }

        NDArray::from_shape_vec([cols, rows], data).unwrap()
    }

    fn matrix_generic_from_rows(rows: usize, cols: usize, values: &[i32]) -> NDArray<i32> {
        let padded_cols = cols + 1;
        let mut data = vec![0_i32; rows * padded_cols];

        for row in 0..rows {
            for col in 0..cols {
                data[row * padded_cols + col] = values[row * cols + col];
            }
        }

        NDArray::from_shape_vec([rows, padded_cols], data).unwrap()
    }

    #[test]
    fn matmul_supports_vector_and_matrix_operands() {
        let lhs_vec = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let rhs_vec = NDArray::from_shape_vec([3], vec![4_i32, 5, 6]).unwrap();
        let matrix = NDArray::from_shape_vec([3, 2], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let left_matrix = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
        let right_matrix = NDArray::from_shape_vec([3, 2], vec![7_i32, 8, 9, 10, 11, 12]).unwrap();

        assert_eq!(matmul(&lhs_vec, &matrix).unwrap().data(), &[22, 28]);
        assert_eq!(matmul(&left_matrix, &rhs_vec).unwrap().data(), &[32, 77]);
        assert_eq!(matmul(&left_matrix, &right_matrix).unwrap().shape(), &[2, 2]);
        assert_eq!(matmul(&left_matrix, &right_matrix).unwrap().data(), &[58, 64, 139, 154]);
    }

    #[test]
    fn matmul_rejects_vector_vector_operands_in_v0_scope() {
        let lhs = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
        let rhs = NDArray::from_shape_vec([3], vec![4_i32, 5, 6]).unwrap();

        assert_eq!(
            matmul(&lhs, &rhs).unwrap_err(),
            AtlasLinalgError::InvalidOperandRank { op: "matmul", left: 1, right: 1 }
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
    fn vector_matrix_kernels_match_row_major_col_major_and_generic_layouts() {
        let lhs = vector_row_major(&[1, 2, 3]);
        let rhs_values = [7_i32, 8, 9, 10, 11, 12];
        let rhs_row_major = matrix_row_major(3, 2, &rhs_values);
        let rhs_col_major_base = matrix_col_major_from_rows(3, 2, &rhs_values);
        let rhs_generic_base = matrix_generic_from_rows(3, 2, &rhs_values);

        let lhs_operand = LinalgOperand::from(&lhs);
        let rhs_row_major_operand = LinalgOperand::from(&rhs_row_major);
        let rhs_col_major_operand = LinalgOperand::from(rhs_col_major_base.view().transpose());
        let rhs_generic_operand =
            LinalgOperand::from(rhs_generic_base.view().slice([0, 0], [3, 2]).unwrap());

        let row_major = matmul_vector_matrix_row_major(
            vector_ref(&lhs_operand),
            matrix_ref(&rhs_row_major_operand),
        );
        let col_major = matmul_vector_matrix_col_major(
            vector_ref(&lhs_operand),
            matrix_ref(&rhs_col_major_operand),
        );
        let generic = matmul_vector_matrix_generic(
            vector_ref(&lhs_operand),
            matrix_ref(&rhs_generic_operand),
        );

        assert_eq!(row_major, col_major);
        assert_eq!(row_major, generic);
    }

    #[test]
    fn matrix_vector_kernels_match_row_major_col_major_and_generic_layouts() {
        let lhs_values = [1_i32, 2, 3, 4, 5, 6];
        let lhs_row_major = matrix_row_major(2, 3, &lhs_values);
        let lhs_col_major_base = matrix_col_major_from_rows(2, 3, &lhs_values);
        let lhs_generic_base = matrix_generic_from_rows(2, 3, &lhs_values);
        let rhs = vector_row_major(&[7, 8, 9]);

        let lhs_row_major_operand = LinalgOperand::from(&lhs_row_major);
        let lhs_col_major_operand = LinalgOperand::from(lhs_col_major_base.view().transpose());
        let lhs_generic_operand =
            LinalgOperand::from(lhs_generic_base.view().slice([0, 0], [2, 3]).unwrap());
        let rhs_operand = LinalgOperand::from(&rhs);

        let row_major = matmul_matrix_vector_row_major(
            matrix_ref(&lhs_row_major_operand),
            vector_ref(&rhs_operand),
        );
        let col_major = matmul_matrix_vector_col_major(
            matrix_ref(&lhs_col_major_operand),
            vector_ref(&rhs_operand),
        );
        let generic = matmul_matrix_vector_generic(
            matrix_ref(&lhs_generic_operand),
            vector_ref(&rhs_operand),
        );

        assert_eq!(row_major, col_major);
        assert_eq!(row_major, generic);
    }

    #[test]
    fn matrix_matrix_kernels_match_specialized_and_generic_layouts() {
        let lhs_values = [1_i32, 2, 3, 4, 5, 6];
        let rhs_values = [7_i32, 8, 9, 10, 11, 12];
        let lhs_row_major = matrix_row_major(2, 3, &lhs_values);
        let rhs_row_major = matrix_row_major(3, 2, &rhs_values);
        let lhs_col_major_base = matrix_col_major_from_rows(2, 3, &lhs_values);
        let rhs_col_major_base = matrix_col_major_from_rows(3, 2, &rhs_values);
        let lhs_generic_base = matrix_generic_from_rows(2, 3, &lhs_values);
        let rhs_generic_base = matrix_generic_from_rows(3, 2, &rhs_values);

        let lhs_row_major_operand = LinalgOperand::from(&lhs_row_major);
        let rhs_row_major_operand = LinalgOperand::from(&rhs_row_major);
        let lhs_col_major_operand = LinalgOperand::from(lhs_col_major_base.view().transpose());
        let rhs_col_major_operand = LinalgOperand::from(rhs_col_major_base.view().transpose());
        let lhs_generic_operand =
            LinalgOperand::from(lhs_generic_base.view().slice([0, 0], [2, 3]).unwrap());
        let rhs_generic_operand =
            LinalgOperand::from(rhs_generic_base.view().slice([0, 0], [3, 2]).unwrap());

        let row_major = matmul_matrix_matrix_row_major(
            matrix_ref(&lhs_row_major_operand),
            matrix_ref(&rhs_row_major_operand),
        );
        let lhs_col_major = matmul_matrix_matrix_lhs_col_major(
            matrix_ref(&lhs_col_major_operand),
            matrix_ref(&rhs_row_major_operand),
        );
        let rhs_col_major = matmul_matrix_matrix_rhs_col_major(
            matrix_ref(&lhs_row_major_operand),
            matrix_ref(&rhs_col_major_operand),
        );
        let generic = matmul_matrix_matrix_generic(
            matrix_ref(&lhs_generic_operand),
            matrix_ref(&rhs_generic_operand),
        );

        assert_eq!(row_major, lhs_col_major);
        assert_eq!(row_major, rhs_col_major);
        assert_eq!(row_major, generic);
    }

    #[test]
    fn matrix_matrix_row_major_blocked_path_matches_generic_for_larger_inputs() {
        let side = 96;
        let lhs_values: Vec<i32> = (0..side * side).map(|index| (index % 7) as i32 - 3).collect();
        let rhs_values: Vec<i32> = (0..side * side).map(|index| (index % 5) as i32 + 1).collect();

        let lhs_row_major = matrix_row_major(side, side, &lhs_values);
        let rhs_row_major = matrix_row_major(side, side, &rhs_values);
        let lhs_generic_base = matrix_generic_from_rows(side, side, &lhs_values);
        let rhs_generic_base = matrix_generic_from_rows(side, side, &rhs_values);

        let lhs_row_major_operand = LinalgOperand::from(&lhs_row_major);
        let rhs_row_major_operand = LinalgOperand::from(&rhs_row_major);
        let lhs_generic_operand =
            LinalgOperand::from(lhs_generic_base.view().slice([0, 0], [side, side]).unwrap());
        let rhs_generic_operand =
            LinalgOperand::from(rhs_generic_base.view().slice([0, 0], [side, side]).unwrap());

        let row_major = matmul_matrix_matrix_row_major(
            matrix_ref(&lhs_row_major_operand),
            matrix_ref(&rhs_row_major_operand),
        );
        let generic = matmul_matrix_matrix_generic(
            matrix_ref(&lhs_generic_operand),
            matrix_ref(&rhs_generic_operand),
        );

        assert_eq!(row_major, generic);
    }

    #[test]
    fn matrix_matrix_row_major_blocked_path_handles_tail_tiles() {
        let side = 97;
        let lhs_values: Vec<i32> = (0..side * side).map(|index| (index % 11) as i32 - 5).collect();
        let rhs_values: Vec<i32> = (0..side * side).map(|index| (index % 13) as i32 - 6).collect();

        let lhs_row_major = matrix_row_major(side, side, &lhs_values);
        let rhs_row_major = matrix_row_major(side, side, &rhs_values);
        let lhs_generic_base = matrix_generic_from_rows(side, side, &lhs_values);
        let rhs_generic_base = matrix_generic_from_rows(side, side, &rhs_values);

        let lhs_row_major_operand = LinalgOperand::from(&lhs_row_major);
        let rhs_row_major_operand = LinalgOperand::from(&rhs_row_major);
        let lhs_generic_operand =
            LinalgOperand::from(lhs_generic_base.view().slice([0, 0], [side, side]).unwrap());
        let rhs_generic_operand =
            LinalgOperand::from(rhs_generic_base.view().slice([0, 0], [side, side]).unwrap());

        let row_major = matmul_matrix_matrix_row_major(
            matrix_ref(&lhs_row_major_operand),
            matrix_ref(&rhs_row_major_operand),
        );
        let generic = matmul_matrix_matrix_generic(
            matrix_ref(&lhs_generic_operand),
            matrix_ref(&rhs_generic_operand),
        );

        assert_eq!(row_major, generic);
    }

    #[test]
    fn matmul_dispatch_stays_serial_for_small_and_medium_inputs() {
        assert!(!should_parallelize_matmul_for_threads(64, 64, 64, 8));
        assert!(!should_parallelize_matmul_for_threads(128, 128, 128, 8));
        assert!(!should_parallelize_matmul_for_threads(128, 128, 128, 16));
        assert!(!should_parallelize_matmul_for_threads(128, 128, 128, 1));
    }

    #[test]
    fn matmul_dispatch_requires_enough_rows_per_thread() {
        assert!(should_parallelize_matmul_for_threads(128, 128, 128, 4));
        assert!(should_parallelize_matmul_for_threads(256, 128, 128, 8));
        assert!(!should_parallelize_matmul_for_threads(255, 128, 128, 8));
    }

    #[test]
    fn matmul_row_major_blocked_dispatch_starts_after_medium_square_inputs() {
        assert!(!should_use_blocked_row_major_matmul_for_dims(64, 64, 64));
        assert!(!should_use_blocked_row_major_matmul_for_dims(80, 80, 80));
        assert!(should_use_blocked_row_major_matmul_for_dims(96, 96, 96));
    }
}
