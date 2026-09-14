use atlas_ndarray::{NDArray, OperandMetadata};

use crate::{
    AtlasMlError, AtlasMlResult,
    core::validation::{validate_finite_feature_values, validate_prediction_feature_inputs},
};

const FIT_OP: &str = "min_max_scaler_fit";
const TRANSFORM_OP: &str = "min_max_scaler_transform";
const INVERSE_TRANSFORM_OP: &str = "min_max_scaler_inverse_transform";

/// Per-feature min-max normalization for rank-2 feature matrices.
#[derive(Clone, Debug, PartialEq)]
pub struct MinMaxScaler {
    minimums: Box<[f64]>,
    maximums: Box<[f64]>,
}

impl MinMaxScaler {
    /// Fits per-feature minimum and maximum values.
    pub fn fit<F>(features: &F) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
    {
        validate_fit_features(features)?;
        validate_finite_feature_values(features, FIT_OP)?;

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

        Ok(Self { minimums: minimums.into(), maximums: maximums.into() })
    }

    /// Returns the fitted per-feature minimums.
    pub fn minimums(&self) -> &[f64] {
        &self.minimums
    }

    /// Returns the fitted per-feature maximums.
    pub fn maximums(&self) -> &[f64] {
        &self.maximums
    }

    /// Normalizes a rank-2 feature matrix relative to the fitted feature ranges.
    ///
    /// Constant features transform to `0.0`.
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
                    0.0
                } else {
                    (feature(features, sample_index, feature_index) - self.minimums[feature_index])
                        / range
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
                    feature(features, sample_index, feature_index) * range
                        + self.minimums[feature_index]
                });
            }
        }

        Ok(NDArray::from_shape_vec([sample_count, feature_count], restored)?)
    }
}

fn validate_fit_features<F>(features: &F) -> AtlasMlResult<()>
where
    F: OperandMetadata<f64> + ?Sized,
{
    if features.ndim() != 2 {
        return Err(AtlasMlError::InvalidInputRank {
            op: FIT_OP,
            expected: "a rank-2 [samples, features] matrix",
            rank: features.ndim(),
        });
    }
    if features.shape()[0] == 0 {
        return Err(AtlasMlError::EmptyInput { op: FIT_OP });
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
