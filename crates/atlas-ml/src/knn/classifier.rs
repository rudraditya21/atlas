use atlas_ndarray::NDArray;

use super::{config::KnnConfig, index::TrainingIndex};
use crate::{
    AtlasMlResult,
    core::validation::{validate_finite_feature_values, validate_supervised_training_inputs},
};

const FIT_OP: &str = "knn_classifier_fit";

pub struct KnnClassifier {
    config: KnnConfig,
    index: TrainingIndex,
    labels: NDArray<usize>,
}

impl KnnClassifier {
    pub fn fit(
        features: NDArray<f64>,
        labels: NDArray<usize>,
        config: KnnConfig,
    ) -> AtlasMlResult<Self> {
        validate_supervised_training_inputs(&features, &labels, FIT_OP)?;
        validate_finite_feature_values(&features, FIT_OP)?;
        config.validate(features.shape()[0])?;

        Ok(Self { config, index: TrainingIndex::new(features, config.search_algorithm()), labels })
    }

    pub const fn config(&self) -> KnnConfig {
        self.config
    }

    pub fn feature_count(&self) -> usize {
        self.index.features().shape()[1]
    }

    pub fn labels(&self) -> &NDArray<usize> {
        &self.labels
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::KnnClassifier;
    use crate::{AtlasMlError, KnnConfig};

    fn features() -> NDArray<f64> {
        NDArray::from_shape_vec([2, 2], vec![0.0_f64, 1.0, 2.0, 3.0]).unwrap()
    }

    #[test]
    fn fits_valid_training_data() {
        let classifier = KnnClassifier::fit(
            features(),
            NDArray::from_shape_vec([2], vec![3_usize, 7]).unwrap(),
            KnnConfig::new(1).unwrap(),
        )
        .unwrap();

        assert_eq!(classifier.config().k(), 1);
        assert_eq!(classifier.feature_count(), 2);
        assert_eq!(classifier.labels().data(), &[3, 7]);
    }

    #[test]
    fn rejects_mismatched_labels() {
        assert_eq!(
            KnnClassifier::fit(
                features(),
                NDArray::from_shape_vec([1], vec![3_usize]).unwrap(),
                KnnConfig::new(1).unwrap(),
            )
            .map(|_| ()),
            Err(AtlasMlError::ShapeMismatch {
                op: "knn_classifier_fit",
                left: vec![2, 2],
                right: vec![1],
                reason: "sample counts must match",
            })
        );
    }

    #[test]
    fn rejects_neighbor_counts_larger_than_training_data() {
        assert_eq!(
            KnnClassifier::fit(
                features(),
                NDArray::from_shape_vec([2], vec![3_usize, 7]).unwrap(),
                KnnConfig::new(3).unwrap(),
            )
            .map(|_| ()),
            Err(AtlasMlError::InvalidArgument {
                op: "knn_config",
                reason: "k must not exceed the number of training samples",
            })
        );
    }

    #[test]
    fn rejects_non_finite_features() {
        let features = NDArray::from_shape_vec([2, 2], vec![0.0_f64, f64::NAN, 2.0, 3.0]).unwrap();

        assert_eq!(
            KnnClassifier::fit(
                features,
                NDArray::from_shape_vec([2], vec![3_usize, 7]).unwrap(),
                KnnConfig::new(1).unwrap(),
            )
            .map(|_| ()),
            Err(AtlasMlError::NonFiniteInput { op: "knn_classifier_fit" })
        );
    }
}
