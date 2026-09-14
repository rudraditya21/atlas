use atlas_ndarray::{NDArray, OperandMetadata};

use crate::{
    AtlasMlError, AtlasMlResult,
    core::validation::{validate_finite_feature_values, validate_prediction_feature_inputs},
};

const FIT_OP: &str = "min_max_scaler_fit";
const FIT_WITH_RANGE_OP: &str = "min_max_scaler_fit_with_range";
const TRANSFORM_OP: &str = "min_max_scaler_transform";
const INVERSE_TRANSFORM_OP: &str = "min_max_scaler_inverse_transform";

/// Per-feature min-max normalization for rank-2 feature matrices.
#[derive(Clone, Debug, PartialEq)]
pub struct MinMaxScaler {
    minimums: Box<[f64]>,
    maximums: Box<[f64]>,
    output_minimum: f64,
    output_maximum: f64,
}

impl MinMaxScaler {
    /// Fits per-feature minimum and maximum values.
    pub fn fit<F>(features: &F) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
    {
        Self::fit_with_range_inner(features, 0.0, 1.0, FIT_OP)
    }

    /// Fits per-feature ranges that transform into `[output_minimum, output_maximum]`.
    pub fn fit_with_range<F>(
        features: &F,
        output_minimum: f64,
        output_maximum: f64,
    ) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
    {
        Self::fit_with_range_inner(features, output_minimum, output_maximum, FIT_WITH_RANGE_OP)
    }

    fn fit_with_range_inner<F>(
        features: &F,
        output_minimum: f64,
        output_maximum: f64,
        op: &'static str,
    ) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
    {
        validate_feature_range(output_minimum, output_maximum, op)?;
        validate_fit_features(features, op)?;
        validate_finite_feature_values(features, op)?;

        let feature_count = features.shape()[1];
        let mut minimums = vec![f64::INFINITY; feature_count];
        let mut maximums = vec![f64::NEG_INFINITY; feature_count];
        for sample_index in 0..features.shape()[0] {
            for feature_index in 0..feature_count {
                let value = feature(features, sample_index, feature_index);
                minimums[feature_index] = minimums[feature_index].min(value);
                maximums[feature_index] = maximums[feature_index].max(value);
            }
        }

        Ok(Self {
            minimums: minimums.into(),
            maximums: maximums.into(),
            output_minimum,
            output_maximum,
        })
    }

    /// Fits this scaler and transforms the same feature matrix.
    pub fn fit_transform<F>(features: &F) -> AtlasMlResult<(Self, NDArray<f64>)>
    where
        F: OperandMetadata<f64> + ?Sized,
    {
        let scaler = Self::fit(features)?;
        let transformed = scaler.transform(features)?;
        Ok((scaler, transformed))
    }

    /// Returns the fitted per-feature minimums.
    pub fn minimums(&self) -> &[f64] {
        &self.minimums
    }

    /// Returns the fitted per-feature maximums.
    pub fn maximums(&self) -> &[f64] {
        &self.maximums
    }

    /// Returns the configured inclusive output interval.
    pub const fn feature_range(&self) -> (f64, f64) {
        (self.output_minimum, self.output_maximum)
    }

    /// Normalizes a rank-2 feature matrix relative to the fitted feature ranges.
    ///
    /// Constant features transform to the configured output minimum.
    pub fn transform<F>(&self, features: &F) -> AtlasMlResult<NDArray<f64>>
    where
        F: OperandMetadata<f64> + ?Sized,
    {
        validate_prediction_feature_inputs(features, self.minimums.len(), TRANSFORM_OP)?;
        validate_finite_feature_values(features, TRANSFORM_OP)?;

        let sample_count = features.shape()[0];
        let feature_count = features.shape()[1];
        let mut transformed = Vec::with_capacity(sample_count * feature_count);
        for sample_index in 0..sample_count {
            for feature_index in 0..feature_count {
                let range = self.maximums[feature_index] - self.minimums[feature_index];
                let value = if range == 0.0 {
                    self.output_minimum
                } else {
                    self.output_minimum
                        + (feature(features, sample_index, feature_index)
                            - self.minimums[feature_index])
                            / range
                            * (self.output_maximum - self.output_minimum)
                };
                transformed.push(value);
            }
        }

        Ok(NDArray::from_shape_vec([sample_count, feature_count], transformed)?)
    }

    /// Restores a rank-2 normalized feature matrix to the fitted feature ranges.
    ///
    /// Constant features restore to their fitted minimum.
    pub fn inverse_transform<F>(&self, features: &F) -> AtlasMlResult<NDArray<f64>>
    where
        F: OperandMetadata<f64> + ?Sized,
    {
        validate_prediction_feature_inputs(features, self.minimums.len(), INVERSE_TRANSFORM_OP)?;
        validate_finite_feature_values(features, INVERSE_TRANSFORM_OP)?;

        let sample_count = features.shape()[0];
        let feature_count = features.shape()[1];
        let mut restored = Vec::with_capacity(sample_count * feature_count);
        for sample_index in 0..sample_count {
            for feature_index in 0..feature_count {
                let range = self.maximums[feature_index] - self.minimums[feature_index];
                restored.push(if range == 0.0 {
                    self.minimums[feature_index]
                } else {
                    (feature(features, sample_index, feature_index) - self.output_minimum)
                        / (self.output_maximum - self.output_minimum)
                        * range
                        + self.minimums[feature_index]
                });
            }
        }

        Ok(NDArray::from_shape_vec([sample_count, feature_count], restored)?)
    }
}

fn validate_fit_features<F>(features: &F, op: &'static str) -> AtlasMlResult<()>
where
    F: OperandMetadata<f64> + ?Sized,
{
    if features.ndim() != 2 {
        return Err(AtlasMlError::InvalidInputRank {
            op,
            expected: "a rank-2 [samples, features] matrix",
            rank: features.ndim(),
        });
    }
    if features.shape()[0] == 0 {
        return Err(AtlasMlError::EmptyInput { op });
    }

    Ok(())
}

fn validate_feature_range(
    output_minimum: f64,
    output_maximum: f64,
    op: &'static str,
) -> AtlasMlResult<()> {
    if !output_minimum.is_finite() || !output_maximum.is_finite() {
        return Err(AtlasMlError::NonFiniteInput { op });
    }
    if output_minimum >= output_maximum {
        return Err(AtlasMlError::InvalidArgument {
            op,
            reason: "feature range minimum must be less than maximum",
        });
    }

    Ok(())
}

fn feature<F>(features: &F, sample_index: usize, feature_index: usize) -> f64
where
    F: OperandMetadata<f64> + ?Sized,
{
    features.data()[features.offset()
        + sample_index * features.strides()[0]
        + feature_index * features.strides()[1]]
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::MinMaxScaler;
    use crate::AtlasMlError;

    #[test]
    fn fits_and_transforms_feature_ranges() {
        let features =
            NDArray::from_shape_vec([3, 2], vec![1.0_f64, 10.0, 3.0, 20.0, 5.0, 40.0]).unwrap();
        let scaler = MinMaxScaler::fit(&features).unwrap();

        assert_eq!(scaler.minimums(), &[1.0, 10.0]);
        assert_eq!(scaler.maximums(), &[5.0, 40.0]);
        assert_close(
            scaler.transform(&features).unwrap().data(),
            &[0.0, 0.0, 0.5, 1.0 / 3.0, 1.0, 1.0],
        );
    }

    #[test]
    fn transforms_constant_features_to_zero() {
        let features = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 5.0, 3.0, 5.0]).unwrap();
        let scaler = MinMaxScaler::fit(&features).unwrap();

        assert_eq!(scaler.transform(&features).unwrap().data(), &[0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn transforms_to_a_configured_feature_range() {
        let features = NDArray::from_shape_vec([3, 1], vec![1.0_f64, 3.0, 5.0]).unwrap();
        let scaler = MinMaxScaler::fit_with_range(&features, -1.0, 1.0).unwrap();

        assert_eq!(scaler.feature_range(), (-1.0, 1.0));
        assert_eq!(scaler.transform(&features).unwrap().data(), &[-1.0, 0.0, 1.0]);
    }

    #[test]
    fn transforms_and_inverses_constant_features_with_a_configured_range() {
        let features = NDArray::from_shape_vec([2, 2], vec![3.0_f64, 5.0, 3.0, 5.0]).unwrap();
        let scaler = MinMaxScaler::fit_with_range(&features, -2.0, 2.0).unwrap();
        let transformed = scaler.transform(&features).unwrap();

        assert_eq!(transformed.data(), &[-2.0, -2.0, -2.0, -2.0]);
        assert_eq!(scaler.inverse_transform(&transformed).unwrap().data(), features.data());
    }

    #[test]
    fn inverses_a_configured_feature_range() {
        let features = NDArray::from_shape_vec([3, 1], vec![1.0_f64, 3.0, 5.0]).unwrap();
        let scaler = MinMaxScaler::fit_with_range(&features, -1.0, 1.0).unwrap();
        let transformed = scaler.transform(&features).unwrap();

        assert_close(scaler.inverse_transform(&transformed).unwrap().data(), features.data());
    }

    #[test]
    fn rejects_invalid_configured_feature_ranges() {
        let features = NDArray::from_shape_vec([1, 1], vec![1.0_f64]).unwrap();

        assert_eq!(
            MinMaxScaler::fit_with_range(&features, 1.0, 1.0),
            Err(AtlasMlError::InvalidArgument {
                op: "min_max_scaler_fit_with_range",
                reason: "feature range minimum must be less than maximum",
            })
        );
        assert_eq!(
            MinMaxScaler::fit_with_range(&features, f64::NAN, 1.0),
            Err(AtlasMlError::NonFiniteInput { op: "min_max_scaler_fit_with_range" })
        );
    }

    #[test]
    fn transforms_logical_views_to_configured_feature_ranges() {
        let source =
            NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let view = source.view().transpose();
        let scaler = MinMaxScaler::fit_with_range(&view, -1.0, 1.0).unwrap();

        assert_eq!(scaler.transform(&view).unwrap().data(), &[-1.0, -1.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn fit_transform_matches_separate_operations_for_views_and_constants() {
        let source =
            NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let view = source.view().transpose();
        let constant = NDArray::from_shape_vec([2, 2], vec![3.0_f64, 5.0, 3.0, 5.0]).unwrap();

        let separate = MinMaxScaler::fit(&view).unwrap();
        let (fitted, transformed) = MinMaxScaler::fit_transform(&view).unwrap();
        assert_eq!(fitted.minimums(), separate.minimums());
        assert_eq!(fitted.maximums(), separate.maximums());
        assert_close(transformed.data(), separate.transform(&view).unwrap().data());

        let separate = MinMaxScaler::fit(&constant).unwrap();
        let (fitted, transformed) = MinMaxScaler::fit_transform(&constant).unwrap();
        assert_eq!(fitted.minimums(), separate.minimums());
        assert_eq!(fitted.maximums(), separate.maximums());
        assert_eq!(transformed.data(), separate.transform(&constant).unwrap().data());
    }

    #[test]
    fn supports_logical_feature_views() {
        let features =
            NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let view = features.view().transpose();
        let scaler = MinMaxScaler::fit(&view).unwrap();

        assert_eq!(scaler.minimums(), &[1.0, 4.0]);
        assert_eq!(scaler.maximums(), &[3.0, 6.0]);
        assert_eq!(scaler.transform(&view).unwrap().data(), &[0.0, 0.0, 0.5, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn transforms_empty_feature_batches() {
        let scaler = MinMaxScaler::fit(
            &NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        )
        .unwrap();
        let transformed = scaler.transform(&NDArray::<f64>::zeros([0, 2]).unwrap()).unwrap();

        assert_eq!(transformed.shape(), &[0, 2]);
        assert!(transformed.data().is_empty());
    }

    #[test]
    fn inverses_transformed_feature_ranges() {
        let features =
            NDArray::from_shape_vec([3, 2], vec![1.0_f64, 10.0, 3.0, 20.0, 5.0, 40.0]).unwrap();
        let scaler = MinMaxScaler::fit(&features).unwrap();
        let normalized = scaler.transform(&features).unwrap();

        assert_close(scaler.inverse_transform(&normalized).unwrap().data(), features.data());
    }

    #[test]
    fn inverses_constant_features_to_their_minimum() {
        let features = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 5.0, 3.0, 5.0]).unwrap();
        let scaler = MinMaxScaler::fit(&features).unwrap();
        let normalized = scaler.transform(&features).unwrap();

        assert_eq!(scaler.inverse_transform(&normalized).unwrap().data(), features.data());
    }

    #[test]
    fn inverses_logical_feature_views() {
        let source =
            NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let view = source.view().transpose();
        let scaler = MinMaxScaler::fit(&view).unwrap();
        let normalized = scaler.transform(&view).unwrap();

        assert_close(
            scaler.inverse_transform(&normalized.view()).unwrap().data(),
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        );
    }

    #[test]
    fn inverses_empty_feature_batches() {
        let scaler = MinMaxScaler::fit(
            &NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        )
        .unwrap();
        let restored = scaler.inverse_transform(&NDArray::<f64>::zeros([0, 2]).unwrap()).unwrap();

        assert_eq!(restored.shape(), &[0, 2]);
        assert!(restored.data().is_empty());
    }

    #[test]
    fn rejects_inverse_transform_width_mismatches() {
        let scaler = MinMaxScaler::fit(
            &NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        )
        .unwrap();
        let features = NDArray::from_shape_vec([1, 1], vec![0.0_f64]).unwrap();

        assert_eq!(
            scaler.inverse_transform(&features).map(|_| ()),
            Err(AtlasMlError::ShapeMismatch {
                op: "min_max_scaler_inverse_transform",
                left: vec![1, 1],
                right: vec![2],
                reason: "feature count must match training data",
            })
        );
    }

    #[test]
    fn rejects_non_finite_features() {
        let finite = NDArray::from_shape_vec([1, 1], vec![1.0_f64]).unwrap();
        let non_finite = NDArray::from_shape_vec([1, 1], vec![f64::INFINITY]).unwrap();
        let scaler = MinMaxScaler::fit(&finite).unwrap();

        assert_eq!(
            MinMaxScaler::fit(&non_finite),
            Err(AtlasMlError::NonFiniteInput { op: "min_max_scaler_fit" })
        );
        assert_eq!(
            scaler.transform(&non_finite).map(|_| ()),
            Err(AtlasMlError::NonFiniteInput { op: "min_max_scaler_transform" })
        );
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
