use crate::{AtlasMlError, AtlasMlResult};

const CONFIG_OP: &str = "perceptron_config";
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

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_LEARNING_RATE, DEFAULT_MAX_ITERATIONS, PerceptronConfig, PerceptronShufflePolicy,
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
}
