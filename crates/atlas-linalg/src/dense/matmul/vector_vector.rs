use atlas_ndarray::{NDArray, Numeric};

use crate::{
    core::{AtlasLinalgError, AtlasLinalgResult, LinalgOperand},
    internal::dense::{dot_kernel, vector_ref},
};

#[allow(dead_code)]
pub(super) fn matmul_vector_vector<T: Numeric>(
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

    Ok(NDArray::from_shape_vec([], vec![dot_kernel(lhs, rhs)])?)
}
