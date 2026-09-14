use atlas_linalg::qr;
use atlas_ndarray::{NDArray, OperandMetadata};

use crate::{
    AtlasMlResult, coefficient_of_determination,
    core::validation::{
        validate_finite_feature_values, validate_finite_target_values,
        validate_prediction_feature_inputs, validate_supervised_training_inputs,
    },
};

const FIT_OP: &str = "linear_regression_fit";
const PREDICT_OP: &str = "linear_regression_predict";

/// Ordinary least-squares linear regression with an intercept term.
pub struct LinearRegression {
    intercept: f64,
    coefficients: NDArray<f64>,
}

impl LinearRegression {
    /// Fits coefficients by QR least squares on an intercept-augmented design matrix.
    pub fn fit<F, T>(features: &F, targets: &T) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
        T: OperandMetadata<f64> + ?Sized,
    {
        validate_supervised_training_inputs(features, targets, FIT_OP)?;
        validate_finite_feature_values(features, FIT_OP)?;

        let target_values = target_values(targets);
        validate_finite_target_values(&target_values, FIT_OP)?;
        let design = design_matrix(features)?;
        let targets = NDArray::from_shape_vec([target_values.len()], target_values)?;
        let solution = qr(&design)?.least_squares(&targets)?;

        Ok(Self {
            intercept: solution.data()[0],
            coefficients: NDArray::from_shape_vec(
                [features.shape()[1]],
                solution.data()[1..].to_vec(),
            )?,
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

    pub fn feature_count(&self) -> usize {
        self.coefficients.shape()[0]
    }

    /// Predicts one target value per query row.
    pub fn predict<Q>(&self, features: &Q) -> AtlasMlResult<NDArray<f64>>
    where
        Q: OperandMetadata<f64> + ?Sized,
    {
        validate_prediction_feature_inputs(features, self.feature_count(), PREDICT_OP)?;
        validate_finite_feature_values(features, PREDICT_OP)?;

        let mut predictions = Vec::with_capacity(features.shape()[0]);
        for sample_index in 0..features.shape()[0] {
            let prediction =
                (0..self.feature_count()).fold(self.intercept, |total, feature_index| {
                    total
                        + feature(features, sample_index, feature_index)
                            * self.coefficients.data()[feature_index]
                });
            predictions.push(prediction);
        }

        Ok(NDArray::from_shape_vec([features.shape()[0]], predictions)?)
    }

    /// Returns the coefficient of determination (R²) for the provided samples.
    pub fn score<F, T>(&self, features: &F, targets: &T) -> AtlasMlResult<f64>
    where
        F: OperandMetadata<f64> + ?Sized,
        T: OperandMetadata<f64> + ?Sized,
    {
        let predictions = self.predict(features)?;
        coefficient_of_determination(targets, &predictions)
    }
}

fn design_matrix<F>(features: &F) -> AtlasMlResult<NDArray<f64>>
where
    F: OperandMetadata<f64> + ?Sized,
{
    let sample_count = features.shape()[0];
    let feature_count = features.shape()[1];
    let mut data = Vec::with_capacity(sample_count * (feature_count + 1));
    for sample_index in 0..sample_count {
        data.push(1.0);
        for feature_index in 0..feature_count {
            data.push(feature(features, sample_index, feature_index));
        }
    }

    Ok(NDArray::from_shape_vec([sample_count, feature_count + 1], data)?)
}

fn feature<F>(features: &F, sample_index: usize, feature_index: usize) -> f64
where
    F: OperandMetadata<f64> + ?Sized,
{
    features.data()[features.offset()
        + sample_index * features.strides()[0]
        + feature_index * features.strides()[1]]
}

fn target_values<T>(targets: &T) -> Vec<f64>
where
    T: OperandMetadata<f64> + ?Sized,
{
    (0..targets.shape()[0])
        .map(|sample_index| targets.data()[targets.offset() + sample_index * targets.strides()[0]])
        .collect()
}

#[cfg(test)]
mod tests {
    use atlas_linalg::AtlasLinalgError;
    use atlas_ndarray::NDArray;

    use super::LinearRegression;
    use crate::AtlasMlError;

    #[test]
    fn fits_and_predicts_simple_linear_data() {
        let features = NDArray::from_shape_vec([3, 1], vec![0.0_f64, 1.0, 2.0]).unwrap();
        let targets = NDArray::from_shape_vec([3], vec![1.0_f64, 3.0, 5.0]).unwrap();
        let model = LinearRegression::fit(&features, &targets).unwrap();
        let queries = NDArray::from_shape_vec([2, 1], vec![3.0_f64, 4.0]).unwrap();

        assert_close(model.intercept(), 1.0);
        assert_close(model.coefficients().data()[0], 2.0);
        assert_close_slice(model.predict(&queries).unwrap().data(), &[7.0, 9.0]);
    }

    #[test]
    fn fits_multiple_features() {
        let features =
            NDArray::from_shape_vec([4, 2], vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .unwrap();
        let targets = NDArray::from_shape_vec([4], vec![1.0_f64, 3.0, -2.0, 0.0]).unwrap();
        let model = LinearRegression::fit(&features, &targets).unwrap();

        assert_close(model.intercept(), 1.0);
        assert_close_slice(model.coefficients().data(), &[2.0, -3.0]);
    }

    #[test]
    fn predicts_from_logical_feature_views() {
        let features = NDArray::from_shape_vec([3, 1], vec![0.0_f64, 1.0, 2.0]).unwrap();
        let targets = NDArray::from_shape_vec([3], vec![1.0_f64, 3.0, 5.0]).unwrap();
        let model = LinearRegression::fit(&features, &targets).unwrap();
        let queries = NDArray::from_shape_vec([1, 2], vec![3.0_f64, 4.0]).unwrap();

        assert_close_slice(model.predict(&queries.view().transpose()).unwrap().data(), &[7.0, 9.0]);
    }

    #[test]
    fn rejects_rank_deficient_features() {
        let features =
            NDArray::from_shape_vec([3, 2], vec![0.0_f64, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let targets = NDArray::from_shape_vec([3], vec![1.0_f64, 3.0, 5.0]).unwrap();

        assert!(matches!(
            LinearRegression::fit(&features, &targets),
            Err(AtlasMlError::Linalg(AtlasLinalgError::RankDeficientMatrix { op: "qr", .. }))
        ));
    }

    #[test]
    fn rejects_invalid_feature_and_target_shapes() {
        let features = NDArray::from_shape_vec([2], vec![0.0_f64, 1.0]).unwrap();
        let targets = NDArray::from_shape_vec([2], vec![1.0_f64, 3.0]).unwrap();
        let valid_features = NDArray::from_shape_vec([2, 1], vec![0.0_f64, 1.0]).unwrap();
        let mismatched_targets = NDArray::from_shape_vec([1], vec![1.0_f64]).unwrap();

        assert_eq!(
            LinearRegression::fit(&features, &targets).map(|_| ()),
            Err(AtlasMlError::InvalidInputRank {
                op: "linear_regression_fit",
                expected: "a rank-2 [samples, features] matrix",
                rank: 1,
            })
        );
        assert_eq!(
            LinearRegression::fit(&valid_features, &mismatched_targets).map(|_| ()),
            Err(AtlasMlError::ShapeMismatch {
                op: "linear_regression_fit",
                left: vec![2, 1],
                right: vec![1],
                reason: "sample counts must match",
            })
        );
    }

    #[test]
    fn scores_perfect_training_data() {
        let features = NDArray::from_shape_vec([3, 1], vec![0.0_f64, 1.0, 2.0]).unwrap();
        let targets = NDArray::from_shape_vec([3], vec![1.0_f64, 3.0, 5.0]).unwrap();
        let model = LinearRegression::fit(&features, &targets).unwrap();

        assert_close(model.score(&features, &targets).unwrap(), 1.0);
    }

    #[test]
    fn scores_held_out_data() {
        let features = NDArray::from_shape_vec([3, 1], vec![0.0_f64, 1.0, 2.0]).unwrap();
        let targets = NDArray::from_shape_vec([3], vec![1.0_f64, 3.0, 5.0]).unwrap();
        let model = LinearRegression::fit(&features, &targets).unwrap();
        let held_out_features = NDArray::from_shape_vec([2, 1], vec![3.0_f64, 4.0]).unwrap();
        let held_out_targets = NDArray::from_shape_vec([2], vec![7.0_f64, 10.0]).unwrap();

        assert_close(model.score(&held_out_features, &held_out_targets).unwrap(), 7.0 / 9.0);
    }

    #[test]
    fn scores_feature_and_target_views() {
        let training_features = NDArray::from_shape_vec([3, 1], vec![0.0_f64, 1.0, 2.0]).unwrap();
        let training_targets = NDArray::from_shape_vec([3], vec![1.0_f64, 3.0, 5.0]).unwrap();
        let model = LinearRegression::fit(&training_features, &training_targets).unwrap();
        let features = NDArray::from_shape_vec([1, 2], vec![3.0_f64, 4.0]).unwrap();
        let targets = NDArray::from_shape_vec([2], vec![7.0_f64, 9.0]).unwrap();

        assert_close(model.score(&features.view().transpose(), &targets.view()).unwrap(), 1.0);
    }

    #[test]
    fn rejects_invalid_score_targets() {
        let features = NDArray::from_shape_vec([2, 1], vec![0.0_f64, 1.0]).unwrap();
        let targets = NDArray::from_shape_vec([2], vec![1.0_f64, 3.0]).unwrap();
        let model = LinearRegression::fit(&features, &targets).unwrap();
        let query = NDArray::from_shape_vec([1, 1], vec![2.0_f64]).unwrap();
        let non_finite_targets = NDArray::from_shape_vec([1], vec![f64::NAN]).unwrap();
        let mismatched_targets = NDArray::from_shape_vec([2], vec![5.0_f64, 7.0]).unwrap();

        assert_eq!(
            model.score(&query, &non_finite_targets).map(|_| ()),
            Err(AtlasMlError::NonFiniteInput { op: "coefficient_of_determination" })
        );
        assert_eq!(
            model.score(&query, &mismatched_targets).map(|_| ()),
            Err(AtlasMlError::ShapeMismatch {
                op: "coefficient_of_determination",
                left: vec![2],
                right: vec![1],
                reason: "target counts must match",
            })
        );
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12);
    }

    fn assert_close_slice(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_close(actual, expected);
        }
    }
}
