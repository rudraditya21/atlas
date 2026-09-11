use atlas_ndarray::NDArray;

use super::{config::KnnConfig, index::TrainingIndex};
use crate::{
    AtlasMlResult,
    core::validation::{
        validate_finite_feature_values, validate_finite_target_values,
        validate_supervised_training_inputs,
    },
};

const FIT_OP: &str = "knn_regressor_fit";

pub struct KnnRegressor {
    config: KnnConfig,
    index: TrainingIndex,
    targets: NDArray<f64>,
}

impl KnnRegressor {
    pub fn fit(
        features: NDArray<f64>,
        targets: NDArray<f64>,
        config: KnnConfig,
    ) -> AtlasMlResult<Self> {
        validate_supervised_training_inputs(&features, &targets, FIT_OP)?;
        validate_finite_feature_values(&features, FIT_OP)?;
        validate_finite_target_values(targets.data(), FIT_OP)?;
        config.validate(features.shape()[0])?;

        Ok(Self { config, index: TrainingIndex::new(features, config.search_algorithm()), targets })
    }

    pub const fn config(&self) -> KnnConfig {
        self.config
    }

    pub fn feature_count(&self) -> usize {
        self.index.features().shape()[1]
    }

    pub fn targets(&self) -> &NDArray<f64> {
        &self.targets
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::KnnRegressor;
    use crate::{AtlasMlError, KnnConfig};

    fn features() -> NDArray<f64> {
        NDArray::from_shape_vec([2, 2], vec![0.0_f64, 1.0, 2.0, 3.0]).unwrap()
    }

    #[test]
    fn fits_valid_training_data() {
        let regressor = KnnRegressor::fit(
            features(),
            NDArray::from_shape_vec([2], vec![1.5_f64, -2.0]).unwrap(),
            KnnConfig::new(1).unwrap(),
        )
        .unwrap();

        assert_eq!(regressor.config().k(), 1);
        assert_eq!(regressor.feature_count(), 2);
        assert_eq!(regressor.targets().data(), &[1.5, -2.0]);
    }

    #[test]
    fn rejects_mismatched_targets() {
        assert_eq!(
            KnnRegressor::fit(
                features(),
                NDArray::from_shape_vec([1], vec![1.5_f64]).unwrap(),
                KnnConfig::new(1).unwrap(),
            )
            .map(|_| ()),
            Err(AtlasMlError::ShapeMismatch {
                op: "knn_regressor_fit",
                left: vec![2, 2],
                right: vec![1],
                reason: "sample counts must match",
            })
        );
    }

    #[test]
    fn rejects_neighbor_counts_larger_than_training_data() {
        assert_eq!(
            KnnRegressor::fit(
                features(),
                NDArray::from_shape_vec([2], vec![1.5_f64, -2.0]).unwrap(),
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
    fn rejects_non_finite_targets() {
        assert_eq!(
            KnnRegressor::fit(
                features(),
                NDArray::from_shape_vec([2], vec![1.5_f64, f64::NAN]).unwrap(),
                KnnConfig::new(1).unwrap(),
            )
            .map(|_| ()),
            Err(AtlasMlError::NonFiniteInput { op: "knn_regressor_fit" })
        );
    }
}
