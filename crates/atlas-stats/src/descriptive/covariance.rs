use atlas_ndarray::Numeric;
use num_traits::ToPrimitive;

use crate::core::{
    AtlasStatsResult, StatsOperand, means, try_for_each_vector_pair_f64, validate_vector_pair,
};

pub fn covariance<'a, T, L, R>(lhs: L, rhs: R) -> AtlasStatsResult<f64>
where
    T: Numeric + ToPrimitive + 'a,
    L: Into<StatsOperand<'a, T>>,
    R: Into<StatsOperand<'a, T>>,
{
    let lhs = lhs.into();
    let rhs = rhs.into();

    validate_vector_pair(&lhs, &rhs, "covariance")?;

    let (lhs_mean, rhs_mean, len) = means(&lhs, &rhs, "covariance")?;
    let mut total = 0.0_f64;

    try_for_each_vector_pair_f64(&lhs, &rhs, "covariance", |left, right| {
        total += (left - lhs_mean) * (right - rhs_mean);
        Ok(())
    })?;

    Ok(total / len as f64)
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasStatsError, covariance};

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() <= 1e-10);
    }

    #[test]
    fn covariance_uses_population_definition() {
        let lhs = NDArray::from_shape_vec([4], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let rhs = NDArray::from_shape_vec([4], vec![2.0_f64, 4.0, 6.0, 8.0]).unwrap();

        assert_close(covariance(&lhs, &rhs).unwrap(), 2.5);
    }

    #[test]
    fn covariance_supports_singleton_vectors() {
        let lhs = NDArray::from_shape_vec([1], vec![5.0_f64]).unwrap();
        let rhs = NDArray::from_shape_vec([1], vec![9.0_f64]).unwrap();

        assert_close(covariance(&lhs, &rhs).unwrap(), 0.0);
    }

    #[test]
    fn covariance_reports_vector_validation_errors() {
        let matrix = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let lhs = NDArray::from_shape_vec([3], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let rhs = NDArray::from_shape_vec([2], vec![1.0_f64, 2.0]).unwrap();

        assert!(matches!(
            covariance(&matrix, &matrix).unwrap_err(),
            AtlasStatsError::InvalidInputRank { op: "covariance", .. }
        ));
        assert!(matches!(
            covariance(&lhs, &rhs).unwrap_err(),
            AtlasStatsError::ShapeMismatch { op: "covariance", .. }
        ));
    }
}
