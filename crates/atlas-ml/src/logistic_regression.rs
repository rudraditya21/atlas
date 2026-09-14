use crate::{AtlasMlError, AtlasMlResult};

const CONFIG_OP: &str = "logistic_regression_config";
const DEFAULT_LEARNING_RATE: f64 = 0.1;
const DEFAULT_MAX_ITERATIONS: usize = 1_000;
const DEFAULT_CONVERGENCE_TOLERANCE: f64 = 1e-6;

/// Configuration for binary logistic regression optimization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogisticRegressionConfig {
    learning_rate: f64,
    max_iterations: usize,
    convergence_tolerance: f64,
}

impl LogisticRegressionConfig {
    /// Creates a configuration with positive, finite optimization hyperparameters.
    pub fn new(
        learning_rate: f64,
        max_iterations: usize,
        convergence_tolerance: f64,
    ) -> AtlasMlResult<Self> {
        validate_positive_finite(learning_rate, "learning rate must be positive")?;
        if max_iterations == 0 {
            return Err(AtlasMlError::InvalidArgument {
                op: CONFIG_OP,
                reason: "iteration limit must be positive",
            });
        }
        validate_positive_finite(convergence_tolerance, "convergence tolerance must be positive")?;

        Ok(Self { learning_rate, max_iterations, convergence_tolerance })
    }

    /// Returns the gradient-descent learning rate.
    pub const fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the maximum number of optimization iterations.
    pub const fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Returns the parameter-change tolerance used for convergence.
    pub const fn convergence_tolerance(&self) -> f64 {
        self.convergence_tolerance
    }
}

impl Default for LogisticRegressionConfig {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            convergence_tolerance: DEFAULT_CONVERGENCE_TOLERANCE,
        }
    }
}

fn validate_positive_finite(value: f64, reason: &'static str) -> AtlasMlResult<()> {
    if !value.is_finite() {
        return Err(AtlasMlError::NonFiniteInput { op: CONFIG_OP });
    }
    if value <= 0.0 {
        return Err(AtlasMlError::InvalidArgument { op: CONFIG_OP, reason });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_CONVERGENCE_TOLERANCE, DEFAULT_LEARNING_RATE, DEFAULT_MAX_ITERATIONS,
        LogisticRegressionConfig,
    };
    use crate::AtlasMlError;

    #[test]
    fn uses_documented_defaults() {
        let config = LogisticRegressionConfig::default();

        assert_eq!(config.learning_rate(), DEFAULT_LEARNING_RATE);
        assert_eq!(config.max_iterations(), DEFAULT_MAX_ITERATIONS);
        assert_eq!(config.convergence_tolerance(), DEFAULT_CONVERGENCE_TOLERANCE);
    }

    #[test]
    fn rejects_invalid_learning_rates() {
        for learning_rate in [0.0, -0.1] {
            assert_eq!(
                LogisticRegressionConfig::new(learning_rate, 1, 1e-6),
                Err(AtlasMlError::InvalidArgument {
                    op: "logistic_regression_config",
                    reason: "learning rate must be positive",
                })
            );
        }
        assert_eq!(
            LogisticRegressionConfig::new(f64::NAN, 1, 1e-6),
            Err(AtlasMlError::NonFiniteInput { op: "logistic_regression_config" })
        );
    }

    #[test]
    fn rejects_zero_iteration_limits() {
        assert_eq!(
            LogisticRegressionConfig::new(0.1, 0, 1e-6),
            Err(AtlasMlError::InvalidArgument {
                op: "logistic_regression_config",
                reason: "iteration limit must be positive",
            })
        );
    }

    #[test]
    fn rejects_invalid_convergence_tolerances() {
        for convergence_tolerance in [0.0, -1e-6] {
            assert_eq!(
                LogisticRegressionConfig::new(0.1, 1, convergence_tolerance),
                Err(AtlasMlError::InvalidArgument {
                    op: "logistic_regression_config",
                    reason: "convergence tolerance must be positive",
                })
            );
        }
        assert_eq!(
            LogisticRegressionConfig::new(0.1, 1, f64::INFINITY),
            Err(AtlasMlError::NonFiniteInput { op: "logistic_regression_config" })
        );
    }
}
