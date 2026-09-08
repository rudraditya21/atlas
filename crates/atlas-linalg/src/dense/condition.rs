use atlas_ndarray::{NDArray, Numeric};
use num_traits::{Float, ToPrimitive};

use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand},
    dense::norm::matrix_frobenius_norm,
    factorization::lu::lu,
    internal::{dense::matrix_ref, factorization::validate_rank_two},
};

/// Estimates `κ_F(A) = ||A||_F * ||A⁻¹||_F` using LU factorization.
///
/// This is a Frobenius-norm estimate, not the spectral (2-norm) condition number.
pub fn condition_number<'a, T, M>(matrix: M) -> AtlasLinalgResult<f64>
where
    T: Numeric + Float + ToPrimitive + 'a,
    M: Into<LinalgOperand<'a, T>>,
{
    let matrix = matrix.into();
    let (rows, columns) = validate_rank_two(&matrix, "condition_number")?;
    if rows != columns {
        return Err(AtlasLinalgError::InvalidInputShape {
            op: "condition_number",
            shape: matrix.shape().to_vec(),
            reason: "condition number requires a square matrix",
        });
    }
    let frobenius_norm = matrix_frobenius_norm(matrix_ref(&matrix))?;
    let factorization = lu(matrix)?;
    let identity = NDArray::eye(rows)?;
    let inverse = factorization.solve(&identity)?;
    let inverse_operand = LinalgOperand::from(&inverse);
    let inverse_norm = matrix_frobenius_norm(matrix_ref(&inverse_operand))?;

    Ok(frobenius_norm * inverse_norm)
}
