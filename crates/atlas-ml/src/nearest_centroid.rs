use atlas_linalg::affine;
use atlas_ndarray::{NDArray, OperandMetadata};

use crate::{
    AtlasMlResult, LabelEncoder,
    core::validation::{
        validate_finite_feature_values, validate_prediction_feature_inputs,
        validate_supervised_training_inputs,
    },
};

const FIT_OP: &str = "nearest_centroid_fit";
const PREDICT_OP: &str = "nearest_centroid_predict";

/// A classifier that assigns each query to its nearest class centroid.
pub struct NearestCentroidClassifier {
    classes: Box<[usize]>,
    centroids: NDArray<f64>,
    centroid_squared_norms: Box<[f64]>,
}

impl NearestCentroidClassifier {
    /// Fits one mean feature vector for each observed class label.
    pub fn fit<F, L>(features: &F, labels: &L) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
        L: OperandMetadata<usize> + ?Sized,
    {
        validate_supervised_training_inputs(features, labels, FIT_OP)?;
        validate_finite_feature_values(features, FIT_OP)?;

        let sample_count = features.shape()[0];
        let feature_count = features.shape()[1];
        let labels = (0..sample_count).map(|index| label(labels, index)).collect::<Vec<_>>();
        let encoder = LabelEncoder::fit(&labels)?;
        let class_indices = encoder.transform(&labels)?;
        let mut centroids = vec![0.0; encoder.classes().len() * feature_count];
        let mut class_counts = vec![0; encoder.classes().len()];

        for (sample_index, class_index) in class_indices.into_iter().enumerate() {
            class_counts[class_index] += 1;
            for feature_index in 0..feature_count {
                centroids[class_index * feature_count + feature_index] +=
                    feature(features, sample_index, feature_index);
            }
        }
        for (class_index, count) in class_counts.into_iter().enumerate() {
            for feature_index in 0..feature_count {
                centroids[class_index * feature_count + feature_index] /= count as f64;
            }
        }

        let centroid_squared_norms = (0..encoder.classes().len())
            .map(|class_index| {
                let start = class_index * feature_count;
                centroids[start..start + feature_count].iter().map(|value| value * value).sum()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Ok(Self {
            classes: encoder.classes().to_vec().into(),
            centroids: NDArray::from_shape_vec(
                [encoder.classes().len(), feature_count],
                centroids,
            )?,
            centroid_squared_norms,
        })
    }

    /// Returns class labels in the centroid-row order.
    pub fn classes(&self) -> &[usize] {
        &self.classes
    }

    /// Returns centroid rows in the class order returned by [`Self::classes`].
    pub fn centroids(&self) -> &NDArray<f64> {
        &self.centroids
    }

    pub fn feature_count(&self) -> usize {
        self.centroids.shape()[1]
    }

    /// Predicts the nearest centroid label for every query row.
    pub fn predict<Q>(&self, queries: &Q) -> AtlasMlResult<NDArray<usize>>
    where
        Q: OperandMetadata<f64> + ?Sized,
    {
        validate_prediction_feature_inputs(queries, self.feature_count(), PREDICT_OP)?;
        validate_finite_feature_values(queries, PREDICT_OP)?;

        let query_count = queries.shape()[0];
        let mut predictions = vec![self.classes[0]; query_count];
        let mut best_scores = vec![f64::NEG_INFINITY; query_count];

        for (class_index, &class) in self.classes.iter().enumerate() {
            let start = class_index * self.feature_count();
            let centroid = &self.centroids.data()[start..start + self.feature_count()];
            let scores =
                affine(queries, centroid, -0.5 * self.centroid_squared_norms[class_index])?;

            for ((prediction, best_score), &score) in
                predictions.iter_mut().zip(&mut best_scores).zip(scores.data())
            {
                if score > *best_score || (score == *best_score && class < *prediction) {
                    *prediction = class;
                    *best_score = score;
                }
            }
        }

        Ok(NDArray::from_shape_vec([query_count], predictions)?)
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

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::NearestCentroidClassifier;
    use crate::AtlasMlError;

    #[test]
    fn predicts_separable_classes() {
        let classifier = NearestCentroidClassifier::fit(
            &NDArray::from_shape_vec([4, 2], vec![0.0_f64, 0.0, 0.2, 0.0, 5.0, 5.0, 5.2, 5.0])
                .unwrap(),
            &NDArray::from_shape_vec([4], vec![1_usize, 1, 4, 4]).unwrap(),
        )
        .unwrap();
        let queries = NDArray::from_shape_vec([2, 2], vec![0.1_f64, 0.0, 5.1, 5.0]).unwrap();

        assert_eq!(classifier.classes(), &[1, 4]);
        assert_eq!(classifier.predict(&queries).unwrap().data(), &[1, 4]);
    }

    #[test]
    fn resolves_equal_centroid_distances_by_lower_label() {
        let classifier = NearestCentroidClassifier::fit(
            &NDArray::from_shape_vec([2, 1], vec![-1.0_f64, 1.0]).unwrap(),
            &NDArray::from_shape_vec([2], vec![2_usize, 1]).unwrap(),
        )
        .unwrap();
        let query = NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap();

        assert_eq!(classifier.predict(&query).unwrap().data(), &[1]);
    }

    #[test]
    fn predicts_empty_query_batches() {
        let classifier = NearestCentroidClassifier::fit(
            &NDArray::from_shape_vec([1, 2], vec![0.0_f64, 0.0]).unwrap(),
            &NDArray::from_shape_vec([1], vec![1_usize]).unwrap(),
        )
        .unwrap();
        let predictions = classifier.predict(&NDArray::<f64>::zeros([0, 2]).unwrap()).unwrap();

        assert_eq!(predictions.shape(), &[0]);
        assert!(predictions.data().is_empty());
    }

    #[test]
    fn rejects_invalid_fitting_input() {
        let features = NDArray::from_shape_vec([2], vec![0.0_f64, 1.0]).unwrap();
        let labels = NDArray::from_shape_vec([2], vec![0_usize, 1]).unwrap();

        assert_eq!(
            NearestCentroidClassifier::fit(&features, &labels).map(|_| ()),
            Err(AtlasMlError::InvalidInputRank {
                op: "nearest_centroid_fit",
                expected: "a rank-2 [samples, features] matrix",
                rank: 1,
            })
        );
    }

    #[test]
    fn supports_logical_feature_and_query_views() {
        let features =
            NDArray::from_shape_vec([2, 3], vec![0.0_f64, 0.2, 5.0, 0.0, 0.0, 5.0]).unwrap();
        let labels = NDArray::from_shape_vec([3], vec![1_usize, 1, 4]).unwrap();
        let classifier =
            NearestCentroidClassifier::fit(&features.view().transpose(), &labels.view()).unwrap();
        let queries = NDArray::from_shape_vec([2, 2], vec![0.1_f64, 5.1, 0.0, 5.0]).unwrap();

        assert_eq!(classifier.predict(&queries.view().transpose()).unwrap().data(), &[1, 4]);
    }
}
