use std::collections::{BTreeMap, BTreeSet};

use atlas_ndarray::{NDArray, OperandMetadata};

use super::{
    config::{KnnConfig, KnnSearchAlgorithm, KnnWeighting},
    index::TrainingIndex,
    metric::SquaredEuclideanDistance,
};
use crate::{
    AtlasMlResult,
    core::{
        row::copy_logical_row,
        validation::{
            validate_finite_feature_values, validate_prediction_feature_inputs,
            validate_prediction_feature_row, validate_supervised_training_inputs,
        },
    },
};

const FIT_OP: &str = "knn_classifier_fit";
const PREDICT_OP: &str = "knn_classifier_predict";
const PREDICT_ONE_OP: &str = "knn_classifier_predict_one";
const PREDICT_PROBA_OP: &str = "knn_classifier_predict_proba";

#[derive(Default)]
struct ClassVote {
    count: usize,
    weight: f64,
    total_distance: f64,
}

pub struct KnnClassifier {
    config: KnnConfig,
    index: TrainingIndex,
    labels: NDArray<usize>,
    classes: Box<[usize]>,
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
        let classes: Box<[usize]> =
            labels.data().iter().copied().collect::<BTreeSet<_>>().into_iter().collect();

        Ok(Self {
            config,
            index: TrainingIndex::new(
                features,
                config.search_algorithm(),
                config.tree_leaf_size(),
            )?,
            labels,
            classes,
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

    pub fn labels(&self) -> &NDArray<usize> {
        &self.labels
    }

    /// Returns the ascending class-label order used by [`Self::predict_proba`].
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Predicts one class per query row.
    ///
    /// Uniform weighting counts neighbor votes. Distance weighting uses inverse
    /// Euclidean distance and considers only exact matches when any exist.
    /// Equal vote weights use the smallest total squared-neighbor distance,
    /// then the lower class label.
    pub fn predict<Q>(&self, queries: &Q) -> AtlasMlResult<NDArray<usize>>
    where
        Q: OperandMetadata<f64> + ?Sized,
    {
        validate_prediction_feature_inputs(queries, self.feature_count(), PREDICT_OP)?;
        validate_finite_feature_values(queries, PREDICT_OP)?;

        let query_count = queries.shape()[0];
        let mut query = vec![0.0; self.feature_count()];
        let mut predictions = Vec::with_capacity(query_count);
        for query_index in 0..query_count {
            copy_logical_row(queries, query_index, &mut query);
            predictions.push(self.predict_one(&query)?);
        }

        Ok(NDArray::from_shape_vec([query_count], predictions)?)
    }

    /// Returns normalized class vote weights for each query row.
    ///
    /// Columns follow the ascending label order returned by [`Self::classes`].
    pub fn predict_proba<Q>(&self, queries: &Q) -> AtlasMlResult<NDArray<f64>>
    where
        Q: OperandMetadata<f64> + ?Sized,
    {
        validate_prediction_feature_inputs(queries, self.feature_count(), PREDICT_PROBA_OP)?;
        validate_finite_feature_values(queries, PREDICT_PROBA_OP)?;

        let query_count = queries.shape()[0];
        let mut query = vec![0.0; self.feature_count()];
        let mut probabilities = Vec::with_capacity(query_count * self.classes.len());
        for query_index in 0..query_count {
            copy_logical_row(queries, query_index, &mut query);
            probabilities.extend(self.probabilities_for(&query)?);
        }

        Ok(NDArray::from_shape_vec([query_count, self.classes.len()], probabilities)?)
    }

    /// Predicts the class for one feature row.
    pub fn predict_one(&self, query: &[f64]) -> AtlasMlResult<usize> {
        validate_prediction_feature_row(query, self.feature_count(), PREDICT_ONE_OP)?;

        Ok(self.class_from_votes(self.votes_for(query)?))
    }

    fn votes_for(&self, query: &[f64]) -> AtlasMlResult<BTreeMap<usize, ClassVote>> {
        let neighbors = self.index.search(query, self.config.k(), &SquaredEuclideanDistance)?;
        let exact_matches = self.config.weighting() == KnnWeighting::Distance
            && neighbors.iter().any(|neighbor| neighbor.distance == 0.0);
        let mut votes = BTreeMap::new();
        for neighbor in neighbors {
            if exact_matches && neighbor.distance != 0.0 {
                continue;
            }

            let vote =
                votes.entry(self.labels.data()[neighbor.index]).or_insert_with(ClassVote::default);
            vote.count += 1;
            vote.weight += match self.config.weighting() {
                KnnWeighting::Uniform => 1.0,
                KnnWeighting::Distance if exact_matches => 1.0,
                KnnWeighting::Distance => neighbor.distance.sqrt().recip(),
            };
            vote.total_distance += neighbor.distance;
        }

        Ok(votes)
    }

    fn class_from_votes(&self, votes: BTreeMap<usize, ClassVote>) -> usize {
        votes
            .into_iter()
            .max_by(|(left_label, left_vote), (right_label, right_vote)| {
                left_vote
                    .weight
                    .total_cmp(&right_vote.weight)
                    .then_with(|| right_vote.total_distance.total_cmp(&left_vote.total_distance))
                    .then_with(|| right_label.cmp(left_label))
            })
            .expect("a fitted classifier always has at least one neighbor")
            .0
    }

    fn probabilities_for(&self, query: &[f64]) -> AtlasMlResult<Vec<f64>> {
        let votes = self.votes_for(query)?;
        let total_weight = votes.values().map(|vote| vote.weight).sum::<f64>();
        let fallback_to_counts = total_weight == 0.0;
        let normalizer = if fallback_to_counts {
            votes.values().map(|vote| vote.count).sum::<usize>() as f64
        } else {
            total_weight
        };

        Ok(self
            .classes
            .iter()
            .map(|class| {
                votes.get(class).map_or(0.0, |vote| {
                    let weight = if fallback_to_counts { vote.count as f64 } else { vote.weight };
                    weight / normalizer
                })
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::KnnClassifier;
    use crate::{AtlasMlError, KnnConfig, KnnSearchAlgorithm, KnnWeighting};

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
        assert_eq!(classifier.selected_search_algorithm(), KnnSearchAlgorithm::BruteForce);
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

    fn classifier(k: usize) -> KnnClassifier {
        KnnClassifier::fit(
            NDArray::from_shape_vec([4, 2], vec![0.0_f64, 0.0, 0.2, 0.0, 5.0, 5.0, 5.2, 5.0])
                .unwrap(),
            NDArray::from_shape_vec([4], vec![0_usize, 0, 1, 1]).unwrap(),
            KnnConfig::new(k).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn predicts_separable_classes() {
        let queries = NDArray::from_shape_vec([2, 2], vec![0.1_f64, 0.0, 5.1, 5.0]).unwrap();

        assert_eq!(classifier(3).predict(&queries).unwrap().data(), &[0, 1]);
    }

    #[test]
    fn probabilities_use_ascending_classes_and_uniform_vote_weights() {
        let classifier = KnnClassifier::fit(
            NDArray::from_shape_vec([3, 1], vec![0.0_f64, 1.0, 2.0]).unwrap(),
            NDArray::from_shape_vec([3], vec![7_usize, 2, 7]).unwrap(),
            KnnConfig::new(3).unwrap(),
        )
        .unwrap();
        let queries = NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap();
        let probabilities = classifier.predict_proba(&queries).unwrap();

        assert_eq!(classifier.classes(), &[2, 7]);
        assert_eq!(probabilities.shape(), &[1, 2]);
        assert_eq!(probabilities.data(), &[1.0 / 3.0, 2.0 / 3.0]);
    }

    #[test]
    fn probabilities_use_distance_weights_and_exact_matches() {
        let distance_weighted = KnnClassifier::fit(
            NDArray::from_shape_vec([3, 1], vec![1.0_f64, 2.0, 0.1]).unwrap(),
            NDArray::from_shape_vec([3], vec![0_usize, 0, 1]).unwrap(),
            KnnConfig::new(3).unwrap().with_weighting(KnnWeighting::Distance),
        )
        .unwrap();
        let exact_matches = KnnClassifier::fit(
            NDArray::from_shape_vec([3, 1], vec![0.0_f64, 0.0, 1.0]).unwrap(),
            NDArray::from_shape_vec([3], vec![5_usize, 2, 5]).unwrap(),
            KnnConfig::new(3).unwrap().with_weighting(KnnWeighting::Distance),
        )
        .unwrap();
        let query = NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap();

        let probabilities = distance_weighted.predict_proba(&query).unwrap();
        assert!((probabilities.data()[0] - 3.0 / 23.0).abs() < 1e-12);
        assert!((probabilities.data()[1] - 20.0 / 23.0).abs() < 1e-12);
        assert_eq!(exact_matches.classes(), &[2, 5]);
        assert_eq!(exact_matches.predict_proba(&query).unwrap().data(), &[0.5, 0.5]);
    }

    #[test]
    fn probabilities_preserve_empty_query_batches() {
        let queries = NDArray::<f64>::zeros([0, 2]).unwrap();
        let probabilities = classifier(1).predict_proba(&queries).unwrap();

        assert_eq!(probabilities.shape(), &[0, 2]);
        assert!(probabilities.data().is_empty());
    }

    #[test]
    fn predicts_with_one_and_all_training_neighbors() {
        let query = NDArray::from_shape_vec([1, 2], vec![5.1_f64, 5.0]).unwrap();
        let all_neighbors = NDArray::from_shape_vec([1, 2], vec![0.1_f64, 0.0]).unwrap();

        assert_eq!(classifier(1).predict(&query).unwrap().data(), &[1]);
        assert_eq!(classifier(4).predict(&all_neighbors).unwrap().data(), &[0]);
    }

    #[test]
    fn predicts_and_validates_single_queries() {
        let model = classifier(1);

        assert_eq!(model.predict_one(&[5.1, 5.0]), Ok(1));
        assert_eq!(
            model.predict_one(&[0.0]),
            Err(AtlasMlError::ShapeMismatch {
                op: "knn_classifier_predict_one",
                left: vec![1],
                right: vec![2],
                reason: "feature count must match training data",
            })
        );
        assert_eq!(
            model.predict_one(&[f64::NAN, 0.0]),
            Err(AtlasMlError::NonFiniteInput { op: "knn_classifier_predict_one" })
        );
    }

    #[test]
    fn predicts_duplicate_labels_for_zero_distance_queries() {
        let classifier = KnnClassifier::fit(
            NDArray::from_shape_vec([3, 1], vec![0.0_f64, 0.0, 2.0]).unwrap(),
            NDArray::from_shape_vec([3], vec![4_usize, 4, 9]).unwrap(),
            KnnConfig::new(2).unwrap(),
        )
        .unwrap();
        let query = NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap();

        assert_eq!(classifier.predict(&query).unwrap().data(), &[4]);
    }

    #[test]
    fn distance_weighting_can_change_the_predicted_class() {
        let features = NDArray::from_shape_vec([3, 1], vec![1.0_f64, 2.0, 0.1]).unwrap();
        let labels = NDArray::from_shape_vec([3], vec![0_usize, 0, 1]).unwrap();
        let uniform =
            KnnClassifier::fit(features.clone(), labels.clone(), KnnConfig::new(3).unwrap())
                .unwrap();
        let distance = KnnClassifier::fit(
            features,
            labels,
            KnnConfig::new(3).unwrap().with_weighting(KnnWeighting::Distance),
        )
        .unwrap();

        assert_eq!(uniform.predict_one(&[0.0]), Ok(0));
        assert_eq!(distance.predict_one(&[0.0]), Ok(1));
    }

    #[test]
    fn distance_weighting_prioritizes_and_stably_resolves_exact_matches() {
        let first = KnnClassifier::fit(
            NDArray::from_shape_vec([3, 1], vec![0.0_f64, 0.0, 0.001]).unwrap(),
            NDArray::from_shape_vec([3], vec![5_usize, 2, 5]).unwrap(),
            KnnConfig::new(3).unwrap().with_weighting(KnnWeighting::Distance),
        )
        .unwrap();
        let second = KnnClassifier::fit(
            NDArray::from_shape_vec([3, 1], vec![0.0_f64, 0.0, 0.001]).unwrap(),
            NDArray::from_shape_vec([3], vec![2_usize, 5, 5]).unwrap(),
            KnnConfig::new(3).unwrap().with_weighting(KnnWeighting::Distance),
        )
        .unwrap();

        assert_eq!(first.predict_one(&[0.0]), Ok(2));
        assert_eq!(second.predict_one(&[0.0]), Ok(2));
    }

    #[test]
    fn predicts_from_logical_query_views() {
        let queries = NDArray::from_shape_vec([2, 2], vec![0.1_f64, 5.1, 0.0, 5.0]).unwrap();

        assert_eq!(classifier(3).predict(&queries.view().transpose()).unwrap().data(), &[0, 1]);
    }

    #[test]
    fn predicts_empty_query_batches() {
        let queries = NDArray::<f64>::zeros([0, 2]).unwrap();
        let predictions = classifier(1).predict(&queries).unwrap();

        assert_eq!(predictions.shape(), &[0]);
        assert!(predictions.data().is_empty());
    }

    #[test]
    fn resolves_equal_vote_counts_by_total_neighbor_distance() {
        let classifier = KnnClassifier::fit(
            NDArray::from_shape_vec([4, 1], vec![1.0_f64, 10.0, 2.0, 3.0]).unwrap(),
            NDArray::from_shape_vec([4], vec![0_usize, 0, 1, 1]).unwrap(),
            KnnConfig::new(4).unwrap(),
        )
        .unwrap();
        let query = NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap();

        assert_eq!(classifier.predict(&query).unwrap().data(), &[1]);
    }

    #[test]
    fn resolves_equal_votes_and_distances_by_lower_label_stably() {
        let first = KnnClassifier::fit(
            NDArray::from_shape_vec([2, 1], vec![-1.0_f64, 1.0]).unwrap(),
            NDArray::from_shape_vec([2], vec![1_usize, 0]).unwrap(),
            KnnConfig::new(2).unwrap(),
        )
        .unwrap();
        let second = KnnClassifier::fit(
            NDArray::from_shape_vec([2, 1], vec![-1.0_f64, 1.0]).unwrap(),
            NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap(),
            KnnConfig::new(2).unwrap(),
        )
        .unwrap();
        let query = NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap();

        assert_eq!(first.predict(&query).unwrap().data(), &[0]);
        assert_eq!(second.predict(&query).unwrap().data(), &[0]);
    }

    #[test]
    fn tree_backend_predictions_match_brute_force() {
        let features =
            NDArray::from_shape_vec([4, 2], vec![0.0_f64, 0.0, 0.2, 0.0, 5.0, 5.0, 5.2, 5.0])
                .unwrap();
        let labels = NDArray::from_shape_vec([4], vec![0_usize, 0, 1, 1]).unwrap();
        let brute_force =
            KnnClassifier::fit(features.clone(), labels.clone(), KnnConfig::new(3).unwrap())
                .unwrap();
        let kd_tree = KnnClassifier::fit(
            features.clone(),
            labels.clone(),
            KnnConfig::new(3).unwrap().with_search_algorithm(KnnSearchAlgorithm::KdTree).unwrap(),
        )
        .unwrap();
        let ball_tree = KnnClassifier::fit(
            features,
            labels,
            KnnConfig::new(3).unwrap().with_search_algorithm(KnnSearchAlgorithm::BallTree).unwrap(),
        )
        .unwrap();
        let queries = NDArray::from_shape_vec([2, 2], vec![0.1_f64, 0.0, 5.1, 5.0]).unwrap();

        assert_eq!(
            kd_tree.predict(&queries).unwrap().data(),
            brute_force.predict(&queries).unwrap().data()
        );
        assert_eq!(
            ball_tree.predict(&queries).unwrap().data(),
            brute_force.predict(&queries).unwrap().data()
        );
    }

    #[test]
    fn reports_explicit_search_backend_selection() {
        for algorithm in [
            KnnSearchAlgorithm::BruteForce,
            KnnSearchAlgorithm::KdTree,
            KnnSearchAlgorithm::BallTree,
        ] {
            let classifier = KnnClassifier::fit(
                features(),
                NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap(),
                KnnConfig::new(1).unwrap().with_search_algorithm(algorithm).unwrap(),
            )
            .unwrap();

            assert_eq!(classifier.selected_search_algorithm(), algorithm);
        }
    }

    #[test]
    fn automatic_backend_predictions_match_brute_force() {
        let small_sample_count = crate::AUTO_BRUTE_FORCE_MAX_SAMPLES;
        let small_features = NDArray::from_shape_vec(
            [small_sample_count, 1],
            (0..small_sample_count).map(|value| value as f64).collect(),
        )
        .unwrap();
        let small_labels = NDArray::from_shape_vec(
            [small_sample_count],
            (0..small_sample_count).map(|value| value % 2).collect(),
        )
        .unwrap();
        let small_brute_force = KnnClassifier::fit(
            small_features.clone(),
            small_labels.clone(),
            KnnConfig::new(3).unwrap(),
        )
        .unwrap();
        let small_automatic = KnnClassifier::fit(
            small_features,
            small_labels,
            KnnConfig::new(3).unwrap().with_search_algorithm(KnnSearchAlgorithm::Auto).unwrap(),
        )
        .unwrap();
        assert_eq!(small_automatic.selected_search_algorithm(), KnnSearchAlgorithm::BruteForce);
        let sample_count = crate::AUTO_BRUTE_FORCE_MAX_SAMPLES + 1;
        let features = NDArray::from_shape_vec(
            [sample_count, 1],
            (0..sample_count).map(|value| value as f64).collect(),
        )
        .unwrap();
        let labels = NDArray::from_shape_vec(
            [sample_count],
            (0..sample_count).map(|value| value % 2).collect(),
        )
        .unwrap();
        let brute_force =
            KnnClassifier::fit(features.clone(), labels.clone(), KnnConfig::new(3).unwrap())
                .unwrap();
        let automatic = KnnClassifier::fit(
            features,
            labels,
            KnnConfig::new(3).unwrap().with_search_algorithm(KnnSearchAlgorithm::Auto).unwrap(),
        )
        .unwrap();
        assert_eq!(automatic.selected_search_algorithm(), KnnSearchAlgorithm::KdTree);
        let small_queries = NDArray::from_shape_vec([1, 1], vec![32.5_f64]).unwrap();
        let queries = NDArray::from_shape_vec([2, 1], vec![0.2_f64, 32.5]).unwrap();

        assert_eq!(
            small_automatic.predict(&small_queries).unwrap().data(),
            small_brute_force.predict(&small_queries).unwrap().data()
        );
        assert_eq!(
            automatic.predict(&queries).unwrap().data(),
            brute_force.predict(&queries).unwrap().data()
        );
    }
}
