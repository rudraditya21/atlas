use atlas_linalg::qr;
use atlas_ndarray::{NDArray, OperandMetadata};

use crate::{
    AtlasMlError, AtlasMlResult, coefficient_of_determination,
    core::validation::{
        validate_finite_feature_values, validate_finite_target_values,
        validate_prediction_feature_inputs, validate_supervised_training_inputs,
    },
};

const CONFIG_OP: &str = "ridge_regression_config";
const FIT_OP: &str = "ridge_regression_fit";
const PREDICT_OP: &str = "ridge_regression_predict";

/// Configuration for ridge regression.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RidgeRegressionConfig {
    l2_regularization: f64,
}

impl RidgeRegressionConfig {
    /// Creates a configuration with a nonnegative L2 regularization strength.
    pub fn new(l2_regularization: f64) -> AtlasMlResult<Self> {
        if !l2_regularization.is_finite() {
            return Err(AtlasMlError::NonFiniteInput { op: CONFIG_OP });
        }
        if l2_regularization < 0.0 {
            return Err(AtlasMlError::InvalidArgument {
                op: CONFIG_OP,
                reason: "l2 regularization strength must be nonnegative",
            });
        }

        Ok(Self { l2_regularization })
    }

    /// Returns the L2 regularization strength.
    pub const fn l2_regularization(&self) -> f64 {
        self.l2_regularization
    }
}

impl Default for RidgeRegressionConfig {
    fn default() -> Self {
        Self { l2_regularization: 0.0 }
    }
}

/// Linear regression with L2 regularization on coefficients but not the intercept.
pub struct RidgeRegression {
    intercept: f64,
    coefficients: NDArray<f64>,
}

impl RidgeRegression {
    /// Fits a ridge-regression model using the supplied configuration.
    pub fn fit<F, T>(
        features: &F,
        targets: &T,
        config: RidgeRegressionConfig,
    ) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
        T: OperandMetadata<f64> + ?Sized,
    {
        validate_supervised_training_inputs(features, targets, FIT_OP)?;
        validate_finite_feature_values(features, FIT_OP)?;

        let target_values = target_values(targets);
        validate_finite_target_values(&target_values, FIT_OP)?;
        let (design, augmented_targets) = regularized_system(features, target_values, config)?;
        let targets = NDArray::from_shape_vec([augmented_targets.len()], augmented_targets)?;
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

    /// Returns the number of input features expected by this model.
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

fn regularized_system<F>(
    features: &F,
    mut targets: Vec<f64>,
    config: RidgeRegressionConfig,
) -> AtlasMlResult<(NDArray<f64>, Vec<f64>)>
where
    F: OperandMetadata<f64> + ?Sized,
{
    let sample_count = features.shape()[0];
    let feature_count = features.shape()[1];
    let mut design = Vec::with_capacity((sample_count + feature_count) * (feature_count + 1));
    for sample_index in 0..sample_count {
        design.push(1.0);
        for feature_index in 0..feature_count {
            design.push(feature(features, sample_index, feature_index));
        }
    }

    let penalty = config.l2_regularization().sqrt();
    for feature_index in 0..feature_count {
        design.push(0.0);
        for column in 0..feature_count {
            design.push(if column == feature_index { penalty } else { 0.0 });
        }
        targets.push(0.0);
    }

    Ok((
        NDArray::from_shape_vec([sample_count + feature_count, feature_count + 1], design)?,
        targets,
    ))
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
    use atlas_ndarray::NDArray;

    use super::{RidgeRegression, RidgeRegressionConfig};
    use crate::AtlasMlError;

    #[test]
    fn accepts_zero_and_positive_regularization() {
        assert_eq!(RidgeRegressionConfig::new(0.0).unwrap().l2_regularization(), 0.0);
        assert_eq!(RidgeRegressionConfig::new(0.25).unwrap().l2_regularization(), 0.25);
    }

    #[test]
    fn rejects_negative_regularization() {
        assert_eq!(
            RidgeRegressionConfig::new(-0.25),
            Err(AtlasMlError::InvalidArgument {
                op: "ridge_regression_config",
                reason: "l2 regularization strength must be nonnegative",
            })
        );
    }

    #[test]
    fn rejects_non_finite_regularization() {
        for l2_regularization in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                RidgeRegressionConfig::new(l2_regularization),
                Err(AtlasMlError::NonFiniteInput { op: "ridge_regression_config" })
            );
        }
    }

    #[test]
    fn shrinks_coefficients() {
        let features = NDArray::from_shape_vec([3, 1], vec![0.0_f64, 1.0, 2.0]).unwrap();
        let targets = NDArray::from_shape_vec([3], vec![1.0_f64, 3.0, 5.0]).unwrap();

        let model =
            RidgeRegression::fit(&features, &targets, RidgeRegressionConfig::new(1.0).unwrap())
                .unwrap();

        assert_close(model.intercept(), 5.0 / 3.0);
        assert_close(model.coefficients().data()[0], 4.0 / 3.0);
    }

    #[test]
    fn leaves_the_intercept_unregularized() {
        let features = NDArray::from_shape_vec([2, 1], vec![10.0_f64, 10.0]).unwrap();
        let targets = NDArray::from_shape_vec([2], vec![1.0_f64, 3.0]).unwrap();

        let model =
            RidgeRegression::fit(&features, &targets, RidgeRegressionConfig::new(1.0).unwrap())
                .unwrap();

        assert_close(model.intercept(), 2.0);
        assert_close(model.coefficients().data()[0], 0.0);
    }

    #[test]
    fn fits_rank_deficient_features_with_regularization() {
        let features =
            NDArray::from_shape_vec([3, 2], vec![0.0_f64, 0.0, 1.0, 1.0, 2.0, 2.0]).unwrap();
        let targets = NDArray::from_shape_vec([3], vec![1.0_f64, 3.0, 5.0]).unwrap();

        let model =
            RidgeRegression::fit(&features, &targets, RidgeRegressionConfig::new(1.0).unwrap())
                .unwrap();

        assert_close_slice(model.predict(&features).unwrap().data(), &[1.4, 3.0, 4.6]);
    }

    #[test]
    fn fits_and_predicts_logical_feature_views() {
        let source = NDArray::from_shape_vec([1, 3], vec![0.0_f64, 1.0, 2.0]).unwrap();
        let features = source.view().transpose();
        let targets = NDArray::from_shape_vec([3], vec![1.0_f64, 3.0, 5.0]).unwrap();
        let model =
            RidgeRegression::fit(&features, &targets, RidgeRegressionConfig::default()).unwrap();
        let queries = NDArray::from_shape_vec([1, 2], vec![3.0_f64, 4.0]).unwrap();

        assert_close_slice(model.predict(&queries.view().transpose()).unwrap().data(), &[7.0, 9.0]);
    }

    #[test]
    fn scores_perfect_training_data() {
        let features = NDArray::from_shape_vec([3, 1], vec![0.0_f64, 1.0, 2.0]).unwrap();
        let targets = NDArray::from_shape_vec([3], vec![1.0_f64, 3.0, 5.0]).unwrap();
        let model =
            RidgeRegression::fit(&features, &targets, RidgeRegressionConfig::default()).unwrap();

        assert_close(model.score(&features, &targets).unwrap(), 1.0);
    }

    #[test]
    fn scores_held_out_data() {
        let features = NDArray::from_shape_vec([3, 1], vec![0.0_f64, 1.0, 2.0]).unwrap();
        let targets = NDArray::from_shape_vec([3], vec![1.0_f64, 3.0, 5.0]).unwrap();
        let model =
            RidgeRegression::fit(&features, &targets, RidgeRegressionConfig::default()).unwrap();
        let held_out_features = NDArray::from_shape_vec([2, 1], vec![3.0_f64, 4.0]).unwrap();
        let held_out_targets = NDArray::from_shape_vec([2], vec![7.0_f64, 10.0]).unwrap();

        assert_close(model.score(&held_out_features, &held_out_targets).unwrap(), 7.0 / 9.0);
    }

    #[test]
    fn scores_logical_feature_and_target_views() {
        let training_features = NDArray::from_shape_vec([3, 1], vec![0.0_f64, 1.0, 2.0]).unwrap();
        let training_targets = NDArray::from_shape_vec([3], vec![1.0_f64, 3.0, 5.0]).unwrap();
        let model = RidgeRegression::fit(
            &training_features,
            &training_targets,
            RidgeRegressionConfig::default(),
        )
        .unwrap();
        let features = NDArray::from_shape_vec([1, 2], vec![3.0_f64, 4.0]).unwrap();
        let targets = NDArray::from_shape_vec([2], vec![7.0_f64, 9.0]).unwrap();

        assert_close(model.score(&features.view().transpose(), &targets.view()).unwrap(), 1.0);
    }

    #[test]
    fn rejects_constant_score_targets() {
        let features = NDArray::from_shape_vec([3, 1], vec![0.0_f64, 1.0, 2.0]).unwrap();
        let targets = NDArray::from_shape_vec([3], vec![4.0_f64, 4.0, 4.0]).unwrap();
        let model =
            RidgeRegression::fit(&features, &targets, RidgeRegressionConfig::new(1.0).unwrap())
                .unwrap();

        assert_eq!(
            model.score(&features, &targets).map(|_| ()),
            Err(AtlasMlError::InvalidArgument {
                op: "coefficient_of_determination",
                reason: "actual targets must have non-zero variance",
            })
        );
    }

    #[test]
    fn rejects_score_target_count_mismatches() {
        let features = NDArray::from_shape_vec([2, 1], vec![0.0_f64, 1.0]).unwrap();
        let targets = NDArray::from_shape_vec([2], vec![1.0_f64, 3.0]).unwrap();
        let model =
            RidgeRegression::fit(&features, &targets, RidgeRegressionConfig::default()).unwrap();
        let mismatched_targets = NDArray::from_shape_vec([1], vec![1.0_f64]).unwrap();

        assert_eq!(
            model.score(&features, &mismatched_targets).map(|_| ()),
            Err(AtlasMlError::ShapeMismatch {
                op: "coefficient_of_determination",
                left: vec![1],
                right: vec![2],
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
