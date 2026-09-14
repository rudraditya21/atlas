use crate::{AtlasMlError, AtlasMlResult};

const CONFIG_OP: &str = "gaussian_naive_bayes_config";
const DEFAULT_VARIANCE_SMOOTHING: f64 = 1e-9;

/// Configuration for Gaussian Naive Bayes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianNaiveBayesConfig {
    variance_smoothing: f64,
}

impl GaussianNaiveBayesConfig {
    /// Creates a configuration with finite, nonnegative variance smoothing.
    pub fn new(variance_smoothing: f64) -> AtlasMlResult<Self> {
        if !variance_smoothing.is_finite() {
            return Err(AtlasMlError::NonFiniteInput { op: CONFIG_OP });
        }
        if variance_smoothing < 0.0 {
            return Err(AtlasMlError::InvalidArgument {
                op: CONFIG_OP,
                reason: "variance smoothing must be nonnegative",
            });
        }

        Ok(Self { variance_smoothing })
    }

    /// Returns the nonnegative value added to estimated feature variances during fitting.
    pub const fn variance_smoothing(&self) -> f64 {
        self.variance_smoothing
    }
}

impl Default for GaussianNaiveBayesConfig {
    fn default() -> Self {
        Self { variance_smoothing: DEFAULT_VARIANCE_SMOOTHING }
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_VARIANCE_SMOOTHING, GaussianNaiveBayesConfig};
    use crate::AtlasMlError;

    #[test]
    fn uses_documented_default_variance_smoothing() {
        assert_eq!(
            GaussianNaiveBayesConfig::default().variance_smoothing(),
            DEFAULT_VARIANCE_SMOOTHING
        );
    }

    #[test]
    fn accepts_zero_variance_smoothing() {
        assert_eq!(GaussianNaiveBayesConfig::new(0.0).unwrap().variance_smoothing(), 0.0);
    }

    #[test]
    fn rejects_negative_variance_smoothing() {
        assert_eq!(
            GaussianNaiveBayesConfig::new(-1e-9),
            Err(AtlasMlError::InvalidArgument {
                op: "gaussian_naive_bayes_config",
                reason: "variance smoothing must be nonnegative",
            })
        );
    }

    #[test]
    fn rejects_non_finite_variance_smoothing() {
        for variance_smoothing in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                GaussianNaiveBayesConfig::new(variance_smoothing),
                Err(AtlasMlError::NonFiniteInput { op: "gaussian_naive_bayes_config" })
            );
        }
    }
}
