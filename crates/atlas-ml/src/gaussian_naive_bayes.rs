use atlas_ndarray::{NDArray, OperandMetadata};

use crate::{
    AtlasMlError, AtlasMlResult, LabelEncoder,
    core::validation::{validate_finite_feature_values, validate_supervised_training_inputs},
};

const CONFIG_OP: &str = "gaussian_naive_bayes_config";
const FIT_OP: &str = "gaussian_naive_bayes_fit";
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

/// Gaussian Naive Bayes parameters estimated independently for each class and feature.
pub struct GaussianNaiveBayes {
    classes: Box<[usize]>,
    class_priors: NDArray<f64>,
    means: NDArray<f64>,
    variances: NDArray<f64>,
}

impl GaussianNaiveBayes {
    /// Fits class priors and per-class Gaussian feature parameters.
    pub fn fit<F, L>(
        features: &F,
        labels: &L,
        config: GaussianNaiveBayesConfig,
    ) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
        L: OperandMetadata<usize> + ?Sized,
    {
        validate_supervised_training_inputs(features, labels, FIT_OP)?;
        validate_finite_feature_values(features, FIT_OP)?;

        let sample_count = features.shape()[0];
        let feature_count = features.shape()[1];
        let labels = label_values(labels);
        let encoder = LabelEncoder::fit(&labels)?;
        let class_indices = encoder.transform(&labels)?;
        let class_count = encoder.classes().len();
        let mut class_counts = vec![0_usize; class_count];
        let mut means = vec![0.0; class_count * feature_count];

        for (sample_index, &class_index) in class_indices.iter().enumerate() {
            class_counts[class_index] += 1;
            for feature_index in 0..feature_count {
                means[class_index * feature_count + feature_index] +=
                    feature(features, sample_index, feature_index);
            }
        }
        for (class_index, &class_count) in class_counts.iter().enumerate() {
            for feature_index in 0..feature_count {
                means[class_index * feature_count + feature_index] /= class_count as f64;
            }
        }

        let mut variances = vec![config.variance_smoothing(); class_count * feature_count];
        for (sample_index, &class_index) in class_indices.iter().enumerate() {
            for feature_index in 0..feature_count {
                let index = class_index * feature_count + feature_index;
                let difference = feature(features, sample_index, feature_index) - means[index];
                variances[index] += difference * difference / class_counts[class_index] as f64;
            }
        }

        Ok(Self {
            classes: encoder.classes().to_vec().into(),
            class_priors: NDArray::from_shape_vec(
                [class_count],
                class_counts
                    .iter()
                    .map(|&class_count| class_count as f64 / sample_count as f64)
                    .collect(),
            )?,
            means: NDArray::from_shape_vec([class_count, feature_count], means)?,
            variances: NDArray::from_shape_vec([class_count, feature_count], variances)?,
        })
    }

    /// Returns class labels in the row order of the fitted parameter matrices.
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Returns empirical class probabilities in the order returned by [`Self::classes`].
    pub fn class_priors(&self) -> &NDArray<f64> {
        &self.class_priors
    }

    /// Returns per-class feature means in the order returned by [`Self::classes`].
    pub fn means(&self) -> &NDArray<f64> {
        &self.means
    }

    /// Returns smoothed per-class feature variances in the order returned by [`Self::classes`].
    pub fn variances(&self) -> &NDArray<f64> {
        &self.variances
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

fn label_values<L>(labels: &L) -> Vec<usize>
where
    L: OperandMetadata<usize> + ?Sized,
{
    (0..labels.shape()[0])
        .map(|index| labels.data()[labels.offset() + index * labels.strides()[0]])
        .collect()
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::{DEFAULT_VARIANCE_SMOOTHING, GaussianNaiveBayes, GaussianNaiveBayesConfig};
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

    #[test]
    fn fits_separable_class_parameters() {
        let features =
            NDArray::from_shape_vec([4, 2], vec![5.0_f64, 5.0, 5.2, 5.0, 0.0, 0.0, 0.2, 0.0])
                .unwrap();
        let labels = NDArray::from_shape_vec([4], vec![4_usize, 4, 1, 1]).unwrap();

        let model = GaussianNaiveBayes::fit(
            &features,
            &labels,
            GaussianNaiveBayesConfig::new(0.1).unwrap(),
        )
        .unwrap();

        assert_eq!(model.classes(), &[1, 4]);
        assert_close_slice(model.class_priors().data(), &[0.5, 0.5]);
        assert_close_slice(model.means().data(), &[0.1, 0.0, 5.1, 5.0]);
        assert_close_slice(model.variances().data(), &[0.11, 0.1, 0.11, 0.1]);
    }

    #[test]
    fn fits_sorted_classes_and_empirical_priors() {
        let features = NDArray::from_shape_vec([3, 1], vec![1.0_f64, 2.0, 3.0]).unwrap();
        let labels = NDArray::from_shape_vec([3], vec![9_usize, 2, 9]).unwrap();

        let model =
            GaussianNaiveBayes::fit(&features, &labels, GaussianNaiveBayesConfig::default())
                .unwrap();

        assert_eq!(model.classes(), &[2, 9]);
        assert_close_slice(model.class_priors().data(), &[1.0 / 3.0, 2.0 / 3.0]);
    }

    #[test]
    fn smooths_constant_feature_variances() {
        let features = NDArray::from_shape_vec([4, 1], vec![2.0_f64; 4]).unwrap();
        let labels = NDArray::from_shape_vec([4], vec![0_usize, 0, 1, 1]).unwrap();

        let model = GaussianNaiveBayes::fit(
            &features,
            &labels,
            GaussianNaiveBayesConfig::new(0.25).unwrap(),
        )
        .unwrap();

        assert_close_slice(model.variances().data(), &[0.25, 0.25]);
    }

    #[test]
    fn rejects_invalid_training_inputs() {
        let features = NDArray::from_shape_vec([2], vec![0.0_f64, 1.0]).unwrap();
        let labels = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();

        assert_eq!(
            GaussianNaiveBayes::fit(&features, &labels, GaussianNaiveBayesConfig::default())
                .map(|_| ()),
            Err(AtlasMlError::InvalidInputRank {
                op: "gaussian_naive_bayes_fit",
                expected: "a rank-2 [samples, features] matrix",
                rank: 1,
            })
        );

        let features = NDArray::from_shape_vec([2, 1], vec![0.0_f64, f64::NAN]).unwrap();
        assert_eq!(
            GaussianNaiveBayes::fit(&features, &labels, GaussianNaiveBayesConfig::default())
                .map(|_| ()),
            Err(AtlasMlError::NonFiniteInput { op: "gaussian_naive_bayes_fit" })
        );
    }

    #[test]
    fn fits_logical_feature_and_label_views() {
        let source = NDArray::from_shape_vec([1, 4], vec![0.0_f64, 1.0, 4.0, 5.0]).unwrap();
        let labels = NDArray::from_shape_vec([1, 4], vec![1_usize, 1, 3, 3]).unwrap();

        let model = GaussianNaiveBayes::fit(
            &source.view().transpose(),
            &labels.view().reshape([4]).unwrap(),
            GaussianNaiveBayesConfig::new(0.1).unwrap(),
        )
        .unwrap();

        assert_eq!(model.classes(), &[1, 3]);
        assert_close_slice(model.means().data(), &[0.5, 4.5]);
    }

    fn assert_close_slice(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
