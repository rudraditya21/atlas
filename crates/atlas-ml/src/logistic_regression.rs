use atlas_ndarray::{NDArray, OperandMetadata};

use crate::{
    AtlasMlError, AtlasMlResult,
    core::validation::{validate_finite_feature_values, validate_supervised_training_inputs},
};

const CONFIG_OP: &str = "logistic_regression_config";
const FIT_OP: &str = "binary_logistic_regression_fit";
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

/// Binary logistic-regression classifier fitted with batch gradient descent.
pub struct BinaryLogisticRegression {
    intercept: f64,
    coefficients: NDArray<f64>,
    iterations: usize,
}

impl BinaryLogisticRegression {
    /// Fits a binary classifier for labels encoded as `0` and `1`.
    pub fn fit<F, L>(
        features: &F,
        labels: &L,
        config: LogisticRegressionConfig,
    ) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
        L: OperandMetadata<usize> + ?Sized,
    {
        validate_supervised_training_inputs(features, labels, FIT_OP)?;
        validate_finite_feature_values(features, FIT_OP)?;
        validate_binary_labels(labels)?;

        let sample_count = features.shape()[0];
        let feature_count = features.shape()[1];
        let mut intercept = 0.0;
        let mut coefficients = vec![0.0; feature_count];
        let mut iterations = 0;

        for iteration in 0..config.max_iterations() {
            let mut intercept_gradient = 0.0;
            let mut coefficient_gradients = vec![0.0; feature_count];
            for sample_index in 0..sample_count {
                let logit = (0..feature_count).fold(intercept, |total, feature_index| {
                    total
                        + feature(features, sample_index, feature_index)
                            * coefficients[feature_index]
                });
                let error = sigmoid(logit) - label(labels, sample_index) as f64;
                intercept_gradient += error;
                for feature_index in 0..feature_count {
                    coefficient_gradients[feature_index] +=
                        error * feature(features, sample_index, feature_index);
                }
            }

            let scale = config.learning_rate() / sample_count as f64;
            let intercept_update = scale * intercept_gradient;
            intercept -= intercept_update;
            let mut maximum_update = intercept_update.abs();
            for feature_index in 0..feature_count {
                let update = scale * coefficient_gradients[feature_index];
                coefficients[feature_index] -= update;
                maximum_update = maximum_update.max(update.abs());
            }
            iterations = iteration + 1;
            if maximum_update <= config.convergence_tolerance() {
                break;
            }
        }

        Ok(Self {
            intercept,
            coefficients: NDArray::from_shape_vec([feature_count], coefficients)?,
            iterations,
        })
    }

    /// Returns the fitted intercept term.
    pub const fn intercept(&self) -> f64 {
        self.intercept
    }

    /// Returns one coefficient per input feature.
    pub fn coefficients(&self) -> &NDArray<f64> {
        &self.coefficients
    }

    /// Returns the number of gradient-descent iterations performed during fitting.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }
}

fn validate_binary_labels<L>(labels: &L) -> AtlasMlResult<()>
where
    L: OperandMetadata<usize> + ?Sized,
{
    if (0..labels.shape()[0]).all(|sample_index| label(labels, sample_index) <= 1) {
        Ok(())
    } else {
        Err(AtlasMlError::InvalidArgument {
            op: FIT_OP,
            reason: "labels must be binary values 0 or 1",
        })
    }
}

fn feature<F>(features: &F, sample_index: usize, feature_index: usize) -> f64
where
    F: OperandMetadata<f64> + ?Sized,
{
    features.data()[features.offset()
        + sample_index * features.strides()[0]
        + feature_index * features.strides()[1]]
}

fn label<L>(labels: &L, sample_index: usize) -> usize
where
    L: OperandMetadata<usize> + ?Sized,
{
    labels.data()[labels.offset() + sample_index * labels.strides()[0]]
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponent = value.exp();
        exponent / (1.0 + exponent)
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
    use atlas_ndarray::NDArray;

    use super::{
        BinaryLogisticRegression, DEFAULT_CONVERGENCE_TOLERANCE, DEFAULT_LEARNING_RATE,
        DEFAULT_MAX_ITERATIONS, LogisticRegressionConfig,
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

    #[test]
    fn fits_separable_data() {
        let features = NDArray::from_shape_vec([4, 1], vec![-2.0_f64, -1.0, 1.0, 2.0]).unwrap();
        let labels = NDArray::from_shape_vec([4], vec![0_usize, 0, 1, 1]).unwrap();
        let config = LogisticRegressionConfig::new(0.5, 1_000, 1e-6).unwrap();

        let model = BinaryLogisticRegression::fit(&features, &labels, config).unwrap();

        assert!(model.coefficients().data()[0] > 0.0);
        assert!(model.intercept().abs() < 1e-12);
    }

    #[test]
    fn stops_when_the_update_converges() {
        let features = NDArray::from_shape_vec([2, 1], vec![0.0_f64, 0.0]).unwrap();
        let labels = NDArray::from_shape_vec([2], vec![0_usize, 0]).unwrap();
        let config = LogisticRegressionConfig::new(0.1, 100, 0.1).unwrap();

        let model = BinaryLogisticRegression::fit(&features, &labels, config).unwrap();

        assert!(model.iterations() < config.max_iterations());
    }

    #[test]
    fn rejects_non_binary_labels() {
        let features = NDArray::from_shape_vec([2, 1], vec![0.0_f64, 1.0]).unwrap();
        let labels = NDArray::from_shape_vec([2], vec![0_usize, 2]).unwrap();

        assert_eq!(
            BinaryLogisticRegression::fit(&features, &labels, LogisticRegressionConfig::default())
                .map(|_| ()),
            Err(AtlasMlError::InvalidArgument {
                op: "binary_logistic_regression_fit",
                reason: "labels must be binary values 0 or 1",
            })
        );
    }

    #[test]
    fn rejects_non_finite_features() {
        let features = NDArray::from_shape_vec([2, 1], vec![0.0_f64, f64::NAN]).unwrap();
        let labels = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();

        assert_eq!(
            BinaryLogisticRegression::fit(&features, &labels, LogisticRegressionConfig::default())
                .map(|_| ()),
            Err(AtlasMlError::NonFiniteInput { op: "binary_logistic_regression_fit" })
        );
    }

    #[test]
    fn fits_logical_feature_and_label_views() {
        let source = NDArray::from_shape_vec([1, 4], vec![-2.0_f64, -1.0, 1.0, 2.0]).unwrap();
        let features = source.view().transpose();
        let labels = NDArray::from_shape_vec([4], vec![0_usize, 0, 1, 1]).unwrap();
        let config = LogisticRegressionConfig::new(0.5, 10, 1e-6).unwrap();

        let model = BinaryLogisticRegression::fit(&features, &labels.view(), config).unwrap();

        assert!(model.coefficients().data()[0] > 0.0);
    }
}
