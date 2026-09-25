use atlas_ndarray::{NDArray, OperandMetadata};

use crate::{
    AtlasMlError, AtlasMlResult,
    core::validation::{
        validate_binary_labels, validate_finite_feature_values, validate_prediction_feature_inputs,
        validate_supervised_training_inputs,
    },
};

const CONFIG_OP: &str = "perceptron_config";
const FIT_OP: &str = "binary_perceptron_fit";
const PREDICT_OP: &str = "binary_perceptron_predict";
const DEFAULT_LEARNING_RATE: f64 = 0.1;
const DEFAULT_MAX_ITERATIONS: usize = 1_000;

/// Sample-order policy for perceptron training iterations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PerceptronShufflePolicy {
    /// Preserve logical training-row order.
    #[default]
    Disabled,
    /// Shuffle each iteration with a deterministic seed.
    Seeded(u64),
}

/// Configuration for binary perceptron training.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PerceptronConfig {
    learning_rate: f64,
    max_iterations: usize,
    shuffle_policy: PerceptronShufflePolicy,
}

impl PerceptronConfig {
    /// Creates a configuration with positive, finite learning rate and iteration limit.
    pub fn new(learning_rate: f64, max_iterations: usize) -> AtlasMlResult<Self> {
        if !learning_rate.is_finite() {
            return Err(AtlasMlError::NonFiniteInput { op: CONFIG_OP });
        }
        if learning_rate <= 0.0 {
            return Err(AtlasMlError::InvalidArgument {
                op: CONFIG_OP,
                reason: "learning rate must be positive",
            });
        }
        if max_iterations == 0 {
            return Err(AtlasMlError::InvalidArgument {
                op: CONFIG_OP,
                reason: "iteration limit must be positive",
            });
        }

        Ok(Self {
            learning_rate,
            max_iterations,
            shuffle_policy: PerceptronShufflePolicy::Disabled,
        })
    }

    /// Returns the update magnitude applied to each misclassified sample.
    pub const fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Returns the maximum number of training iterations.
    pub const fn max_iterations(&self) -> usize {
        self.max_iterations
    }

    /// Returns the training-row shuffle policy.
    pub const fn shuffle_policy(&self) -> PerceptronShufflePolicy {
        self.shuffle_policy
    }

    /// Returns a configuration with the specified deterministic shuffle policy.
    pub const fn with_shuffle_policy(mut self, shuffle_policy: PerceptronShufflePolicy) -> Self {
        self.shuffle_policy = shuffle_policy;
        self
    }
}

impl Default for PerceptronConfig {
    fn default() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            shuffle_policy: PerceptronShufflePolicy::Disabled,
        }
    }
}

/// Linear binary classifier trained with the perceptron update rule.
pub struct BinaryPerceptron {
    intercept: f64,
    coefficients: NDArray<f64>,
    iterations: usize,
    converged: bool,
}

impl BinaryPerceptron {
    /// Fits a binary classifier for labels encoded as `0` and `1`.
    pub fn fit<F, L>(features: &F, labels: &L, config: PerceptronConfig) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
        L: OperandMetadata<usize> + ?Sized,
    {
        validate_supervised_training_inputs(features, labels, FIT_OP)?;
        validate_finite_feature_values(features, FIT_OP)?;
        validate_binary_labels(labels, FIT_OP)?;

        let sample_count = features.shape()[0];
        let feature_count = features.shape()[1];
        let mut intercept = 0.0;
        let mut coefficients = vec![0.0; feature_count];
        let mut indices = (0..sample_count).collect::<Vec<_>>();
        let mut iterations = 0;
        let mut converged = false;

        for iteration in 0..config.max_iterations() {
            if let PerceptronShufflePolicy::Seeded(seed) = config.shuffle_policy() {
                indices = (0..sample_count).collect();
                shuffle(&mut indices, seed.wrapping_add(iteration as u64));
            }

            let mut mistakes = 0;
            for &sample_index in &indices {
                let score = (0..feature_count).fold(intercept, |total, feature_index| {
                    total
                        + feature(features, sample_index, feature_index)
                            * coefficients[feature_index]
                });
                let target = label(labels, sample_index);
                if classify(score) != target {
                    let direction = if target == 1 { 1.0 } else { -1.0 };
                    intercept += config.learning_rate() * direction;
                    for (feature_index, coefficient) in coefficients.iter_mut().enumerate() {
                        *coefficient += config.learning_rate()
                            * direction
                            * feature(features, sample_index, feature_index);
                    }
                    mistakes += 1;
                }
            }
            iterations = iteration + 1;
            if mistakes == 0 {
                converged = true;
                break;
            }
        }

        Ok(Self {
            intercept,
            coefficients: NDArray::from_shape_vec([feature_count], coefficients)?,
            iterations,
            converged,
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

    /// Returns the number of input features expected by this model.
    pub fn feature_count(&self) -> usize {
        self.coefficients.shape()[0]
    }

    /// Returns the number of training iterations completed.
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns whether an iteration completed without classification mistakes.
    pub const fn converged(&self) -> bool {
        self.converged
    }

    /// Predicts one binary class per query row.
    pub fn predict<Q>(&self, queries: &Q) -> AtlasMlResult<NDArray<usize>>
    where
        Q: OperandMetadata<f64> + ?Sized,
    {
        validate_prediction_feature_inputs(queries, self.feature_count(), PREDICT_OP)?;
        validate_finite_feature_values(queries, PREDICT_OP)?;

        let predictions = (0..queries.shape()[0])
            .map(|query_index| {
                classify((0..self.feature_count()).fold(self.intercept, |total, feature_index| {
                    total
                        + feature(queries, query_index, feature_index)
                            * self.coefficients.data()[feature_index]
                }))
            })
            .collect();

        Ok(NDArray::from_shape_vec([queries.shape()[0]], predictions)?)
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

fn classify(score: f64) -> usize {
    if score >= 0.0 { 1 } else { 0 }
}

fn shuffle(indices: &mut [usize], seed: u64) {
    let mut state = seed;
    for upper_bound in (2..=indices.len()).rev() {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        indices.swap(upper_bound - 1, (state % upper_bound as u64) as usize);
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::{
        BinaryPerceptron, DEFAULT_LEARNING_RATE, DEFAULT_MAX_ITERATIONS, PerceptronConfig,
        PerceptronShufflePolicy,
    };
    use crate::AtlasMlError;

    #[test]
    fn uses_documented_defaults() {
        let config = PerceptronConfig::default();

        assert_eq!(config.learning_rate(), DEFAULT_LEARNING_RATE);
        assert_eq!(config.max_iterations(), DEFAULT_MAX_ITERATIONS);
        assert_eq!(config.shuffle_policy(), PerceptronShufflePolicy::Disabled);
    }

    #[test]
    fn supports_deterministic_shuffle_policy() {
        let config = PerceptronConfig::new(0.25, 10)
            .unwrap()
            .with_shuffle_policy(PerceptronShufflePolicy::Seeded(42));

        assert_eq!(config.shuffle_policy(), PerceptronShufflePolicy::Seeded(42));
    }

    #[test]
    fn rejects_invalid_learning_rates() {
        for learning_rate in [0.0, -0.1] {
            assert_eq!(
                PerceptronConfig::new(learning_rate, 1),
                Err(AtlasMlError::InvalidArgument {
                    op: "perceptron_config",
                    reason: "learning rate must be positive",
                })
            );
        }
        for learning_rate in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                PerceptronConfig::new(learning_rate, 1),
                Err(AtlasMlError::NonFiniteInput { op: "perceptron_config" })
            );
        }
    }

    #[test]
    fn rejects_zero_iteration_limit() {
        assert_eq!(
            PerceptronConfig::new(0.1, 0),
            Err(AtlasMlError::InvalidArgument {
                op: "perceptron_config",
                reason: "iteration limit must be positive",
            })
        );
    }

    #[test]
    fn fits_and_predicts_separable_data() {
        let features =
            NDArray::from_shape_vec([4, 2], vec![-1.0_f64, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
                .unwrap();
        let labels = NDArray::from_shape_vec([4], vec![0_usize, 0, 1, 1]).unwrap();
        let config = PerceptronConfig::new(1.0, 20)
            .unwrap()
            .with_shuffle_policy(PerceptronShufflePolicy::Seeded(7));

        let model = BinaryPerceptron::fit(&features, &labels, config).unwrap();

        assert!(model.converged());
        assert_eq!(model.predict(&features).unwrap().data(), labels.data());
    }

    #[test]
    fn reports_iteration_limit_on_nonseparable_data() {
        let features =
            NDArray::from_shape_vec([4, 2], vec![0.0_f64, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
                .unwrap();
        let labels = NDArray::from_shape_vec([4], vec![0_usize, 1, 1, 0]).unwrap();
        let config = PerceptronConfig::new(1.0, 1).unwrap();

        let model = BinaryPerceptron::fit(&features, &labels, config).unwrap();

        assert_eq!(model.iterations(), 1);
        assert!(!model.converged());
    }

    #[test]
    fn rejects_non_binary_training_labels() {
        let features = NDArray::from_shape_vec([2, 1], vec![0.0_f64, 1.0]).unwrap();
        let labels = NDArray::from_shape_vec([2], vec![0_usize, 2]).unwrap();

        assert_eq!(
            BinaryPerceptron::fit(&features, &labels, PerceptronConfig::default()).map(|_| ()),
            Err(AtlasMlError::InvalidArgument {
                op: "binary_perceptron_fit",
                reason: "labels must be binary values 0 or 1",
            })
        );
    }

    #[test]
    fn fits_and_predicts_logical_views() {
        let source =
            NDArray::from_shape_vec([2, 4], vec![-1.0_f64, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
                .unwrap();
        let labels = NDArray::from_shape_vec([1, 4], vec![0_usize, 0, 1, 1]).unwrap();
        let features = source.view().transpose();
        let labels = labels.view().reshape([4]).unwrap();

        let model =
            BinaryPerceptron::fit(&features, &labels, PerceptronConfig::new(1.0, 20).unwrap())
                .unwrap();

        assert_eq!(model.predict(&features).unwrap().data(), &[0, 0, 1, 1]);
    }
}
