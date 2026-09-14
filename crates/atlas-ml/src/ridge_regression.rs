use crate::{AtlasMlError, AtlasMlResult};

const CONFIG_OP: &str = "ridge_regression_config";

/// Configuration for ridge regression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RidgeRegressionConfig {
    l2_regularization: f64,
}

impl RidgeRegressionConfig {
    /// Creates a configuration with a nonnegative L2 regularization strength.
    pub fn new(l2_regularization: f64) -> AtlasMlResult<Self> {
        if !l2_regularization.is_finite() {
            return Err(AtlasMlError::NonFiniteInput { op: CONFIG_OP });
        }
        if l2_regularization < 0.0 {
            return Err(AtlasMlError::InvalidArgument {
                op: CONFIG_OP,
                reason: "l2 regularization strength must be nonnegative",
            });
        }

        Ok(Self { l2_regularization })
    }

    /// Returns the L2 regularization strength.
    pub const fn l2_regularization(&self) -> f64 {
        self.l2_regularization
    }
}

impl Default for RidgeRegressionConfig {
    fn default() -> Self {
        Self { l2_regularization: 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::RidgeRegressionConfig;
    use crate::AtlasMlError;

    #[test]
    fn accepts_zero_and_positive_regularization() {
        assert_eq!(RidgeRegressionConfig::new(0.0).unwrap().l2_regularization(), 0.0);
        assert_eq!(RidgeRegressionConfig::new(0.25).unwrap().l2_regularization(), 0.25);
    }

    #[test]
    fn rejects_negative_regularization() {
        assert_eq!(
            RidgeRegressionConfig::new(-0.25),
            Err(AtlasMlError::InvalidArgument {
                op: "ridge_regression_config",
                reason: "l2 regularization strength must be nonnegative",
            })
        );
    }

    #[test]
    fn rejects_non_finite_regularization() {
        for l2_regularization in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                RidgeRegressionConfig::new(l2_regularization),
                Err(AtlasMlError::NonFiniteInput { op: "ridge_regression_config" })
            );
        }
    }
}
