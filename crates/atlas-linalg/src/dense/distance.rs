use crate::{AtlasLinalgError, AtlasLinalgResult, internal::simd};

/// Returns the squared Euclidean distance between two vectors.
pub fn squared_euclidean_distance(lhs: &[f64], rhs: &[f64]) -> AtlasLinalgResult<f64> {
    if lhs.len() != rhs.len() {
        return Err(AtlasLinalgError::ShapeMismatch {
            op: "squared_euclidean_distance",
            left: vec![lhs.len()],
            right: vec![rhs.len()],
            reason: "vector lengths must match",
        });
    }

    Ok(simd::squared_euclidean_f64(lhs, rhs))
}
