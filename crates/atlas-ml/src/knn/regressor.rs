use atlas_ndarray::{NDArray, OperandMetadata};

use super::{
    config::{KnnConfig, KnnSearchAlgorithm, KnnWeighting},
    index::TrainingIndex,
    metric::SquaredEuclideanDistance,
    neighbor::Neighbor,
    row::copy_row,
};
use crate::{
    AtlasMlResult,
    core::validation::{
        validate_finite_feature_values, validate_finite_target_values,
        validate_prediction_feature_inputs, validate_prediction_feature_row,
        validate_supervised_training_inputs,
    },
};

const FIT_OP: &str = "knn_regressor_fit";
const PREDICT_OP: &str = "knn_regressor_predict";
const PREDICT_ONE_OP: &str = "knn_regressor_predict_one";

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

        Ok(Self {
            config,
            index: TrainingIndex::new(
                features,
                config.search_algorithm(),
                config.tree_leaf_size(),
            )?,
            targets,
        })
    }

    pub const fn config(&self) -> KnnConfig {
        self.config
    }

    /// Returns the backend selected when this model was fitted.
    pub fn selected_search_algorithm(&self) -> KnnSearchAlgorithm {
        self.index.search_algorithm()
    }

    pub fn feature_count(&self) -> usize {
        self.index.features().shape()[1]
    }

    pub fn targets(&self) -> &NDArray<f64> {
        &self.targets
    }

    pub fn predict<Q>(&self, queries: &Q) -> AtlasMlResult<NDArray<f64>>
    where
        Q: OperandMetadata<f64> + ?Sized,
    {
        validate_prediction_feature_inputs(queries, self.feature_count(), PREDICT_OP)?;
        validate_finite_feature_values(queries, PREDICT_OP)?;

        let query_count = queries.shape()[0];
        let mut query = vec![0.0; self.feature_count()];
        let mut predictions = Vec::with_capacity(query_count);
        for query_index in 0..query_count {
            copy_row(queries, query_index, &mut query);
            predictions.push(self.predict_one(&query)?);
        }

        Ok(NDArray::from_shape_vec([query_count], predictions)?)
    }

    /// Predicts the target for one feature row.
    pub fn predict_one(&self, query: &[f64]) -> AtlasMlResult<f64> {
        validate_prediction_feature_row(query, self.feature_count(), PREDICT_ONE_OP)?;

        let neighbors = self.index.search(query, self.config.k(), &SquaredEuclideanDistance)?;

        Ok(match self.config.weighting() {
            KnnWeighting::Uniform => mean_targets(&neighbors, self.targets.data()),
            KnnWeighting::Distance => distance_weighted_mean(&neighbors, self.targets.data()),
        })
    }
}

fn mean_targets(neighbors: &[Neighbor], targets: &[f64]) -> f64 {
    neighbors.iter().map(|neighbor| targets[neighbor.index]).sum::<f64>() / neighbors.len() as f64
}

fn distance_weighted_mean(neighbors: &[Neighbor], targets: &[f64]) -> f64 {
    let mut exact_total = 0.0;
    let mut exact_count = 0;
    for neighbor in neighbors {
        if neighbor.distance == 0.0 {
            exact_total += targets[neighbor.index];
            exact_count += 1;
        }
    }
    if exact_count != 0 {
        return exact_total / exact_count as f64;
    }

    let (weighted_total, total_weight) =
        neighbors.iter().fold((0.0, 0.0), |(total, weight), neighbor| {
            let neighbor_weight = neighbor.distance.sqrt().recip();
            (total + neighbor_weight * targets[neighbor.index], weight + neighbor_weight)
        });
    if total_weight == 0.0 {
        mean_targets(neighbors, targets)
    } else {
        weighted_total / total_weight
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::KnnRegressor;
    use crate::{AtlasMlError, KnnConfig, KnnSearchAlgorithm, KnnWeighting};

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
        assert_eq!(regressor.selected_search_algorithm(), KnnSearchAlgorithm::BruteForce);
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

    fn regressor(k: usize) -> KnnRegressor {
        KnnRegressor::fit(
            NDArray::from_shape_vec([3, 1], vec![0.0_f64, 2.0, 4.0]).unwrap(),
            NDArray::from_shape_vec([3], vec![0.0_f64, 2.0, 10.0]).unwrap(),
            KnnConfig::new(k).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn predicts_exact_training_targets() {
        let query = NDArray::from_shape_vec([1, 1], vec![2.0_f64]).unwrap();

        assert_eq!(regressor(1).predict(&query).unwrap().data(), &[2.0]);
    }

    #[test]
    fn predicts_and_validates_single_queries() {
        let model = regressor(1);

        assert_eq!(model.predict_one(&[2.0]), Ok(2.0));
        assert_eq!(
            model.predict_one(&[2.0, 3.0]),
            Err(AtlasMlError::ShapeMismatch {
                op: "knn_regressor_predict_one",
                left: vec![2],
                right: vec![1],
                reason: "feature count must match training data",
            })
        );
        assert_eq!(
            model.predict_one(&[f64::INFINITY]),
            Err(AtlasMlError::NonFiniteInput { op: "knn_regressor_predict_one" })
        );
    }

    #[test]
    fn averages_selected_neighbor_targets() {
        let query = NDArray::from_shape_vec([1, 1], vec![1.0_f64]).unwrap();

        assert_eq!(regressor(2).predict(&query).unwrap().data(), &[1.0]);
    }

    #[test]
    fn predicts_from_logical_query_views() {
        let queries = NDArray::from_shape_vec([1, 2], vec![1.0_f64, 3.0]).unwrap();

        assert_eq!(regressor(2).predict(&queries.view().transpose()).unwrap().data(), &[1.0, 6.0]);
    }

    #[test]
    fn predicts_empty_query_batches() {
        let queries = NDArray::<f64>::zeros([0, 1]).unwrap();
        let predictions = regressor(1).predict(&queries).unwrap();

        assert_eq!(predictions.shape(), &[0]);
        assert!(predictions.data().is_empty());
    }

    #[test]
    fn predicts_with_minimum_and_maximum_neighbor_counts() {
        let query = NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap();

        assert_eq!(regressor(1).predict(&query).unwrap().data(), &[0.0]);
        assert_eq!(regressor(3).predict(&query).unwrap().data(), &[4.0]);
    }

    #[test]
    fn predicts_known_distance_weighted_means() {
        let regressor = KnnRegressor::fit(
            NDArray::from_shape_vec([2, 1], vec![1.0_f64, 3.0]).unwrap(),
            NDArray::from_shape_vec([2], vec![0.0_f64, 12.0]).unwrap(),
            KnnConfig::new(2).unwrap().with_weighting(KnnWeighting::Distance),
        )
        .unwrap();
        let query = NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap();

        assert_eq!(regressor.predict(&query).unwrap().data(), &[3.0]);
    }

    #[test]
    fn averages_repeated_exact_matches_before_other_neighbors() {
        let regressor = KnnRegressor::fit(
            NDArray::from_shape_vec([3, 1], vec![0.0_f64, 0.0, 2.0]).unwrap(),
            NDArray::from_shape_vec([3], vec![2.0_f64, 4.0, 100.0]).unwrap(),
            KnnConfig::new(3).unwrap().with_weighting(KnnWeighting::Distance),
        )
        .unwrap();
        let query = NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap();

        assert_eq!(regressor.predict(&query).unwrap().data(), &[3.0]);
    }

    #[test]
    fn tree_backend_predictions_match_brute_force() {
        let features = NDArray::from_shape_vec([3, 1], vec![0.0_f64, 2.0, 4.0]).unwrap();
        let targets = NDArray::from_shape_vec([3], vec![0.0_f64, 2.0, 10.0]).unwrap();
        let brute_force = KnnRegressor::fit(
            features.clone(),
            targets.clone(),
            KnnConfig::new(2).unwrap().with_weighting(KnnWeighting::Distance),
        )
        .unwrap();
        let kd_tree = KnnRegressor::fit(
            features.clone(),
            targets.clone(),
            KnnConfig::new(2)
                .unwrap()
                .with_weighting(KnnWeighting::Distance)
                .with_search_algorithm(KnnSearchAlgorithm::KdTree)
                .unwrap(),
        )
        .unwrap();
        let ball_tree = KnnRegressor::fit(
            features,
            targets,
            KnnConfig::new(2)
                .unwrap()
                .with_weighting(KnnWeighting::Distance)
                .with_search_algorithm(KnnSearchAlgorithm::BallTree)
                .unwrap(),
        )
        .unwrap();
        let queries = NDArray::from_shape_vec([2, 1], vec![1.0_f64, 3.0]).unwrap();

        assert_eq!(
            kd_tree.predict(&queries).unwrap().data(),
            brute_force.predict(&queries).unwrap().data()
        );
        assert_eq!(
            ball_tree.predict(&queries).unwrap().data(),
            brute_force.predict(&queries).unwrap().data()
        );
    }
}
