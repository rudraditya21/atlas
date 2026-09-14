use atlas_ndarray::{NDArray, OperandMetadata};

use crate::{
    AtlasMlError, AtlasMlResult,
    core::validation::{validate_finite_feature_values, validate_prediction_feature_inputs},
};

const FIT_OP: &str = "standard_scaler_fit";
const TRANSFORM_OP: &str = "standard_scaler_transform";

/// Per-feature population-standard-deviation scaling for rank-2 feature matrices.
#[derive(Clone, Debug, PartialEq)]
pub struct StandardScaler {
    means: Box<[f64]>,
    scales: Box<[f64]>,
}

impl StandardScaler {
    /// Fits feature means and population standard deviations.
    ///
    /// Constant features retain a unit scale and transform to zero.
    pub fn fit<F>(features: &F) -> AtlasMlResult<Self>
    where
        F: OperandMetadata<f64> + ?Sized,
    {
        validate_fit_features(features)?;
        validate_finite_feature_values(features, FIT_OP)?;

        let sample_count = features.shape()[0];
        let feature_count = features.shape()[1];
        let mut means = vec![0.0; feature_count];
        let mut sum_squares = vec![0.0; feature_count];
        for sample_index in 0..sample_count {
            let count = (sample_index + 1) as f64;
            for feature_index in 0..feature_count {
                let value = feature(features, sample_index, feature_index);
                let delta = value - means[feature_index];
                means[feature_index] += delta / count;
                sum_squares[feature_index] += delta * (value - means[feature_index]);
            }
        }
        let scales = sum_squares
            .into_iter()
            .map(|sum| {
                let scale = (sum / sample_count as f64).sqrt();
                if scale == 0.0 { 1.0 } else { scale }
            })
            .collect();

        Ok(Self { means: means.into(), scales })
    }

    /// Returns the fitted per-feature means.
    pub fn means(&self) -> &[f64] {
        &self.means
    }

    /// Returns the fitted per-feature scales.
    ///
    /// Constant features have a scale of `1.0`.
    pub fn scales(&self) -> &[f64] {
        &self.scales
    }

    /// Transforms a rank-2 feature matrix into standardized values.
    pub fn transform<F>(&self, features: &F) -> AtlasMlResult<NDArray<f64>>
    where
        F: OperandMetadata<f64> + ?Sized,
    {
        validate_prediction_feature_inputs(features, self.means.len(), TRANSFORM_OP)?;
        validate_finite_feature_values(features, TRANSFORM_OP)?;

        let sample_count = features.shape()[0];
        let feature_count = features.shape()[1];
        let mut transformed = Vec::with_capacity(sample_count * feature_count);
        for sample_index in 0..sample_count {
            for feature_index in 0..feature_count {
                transformed.push(
                    (feature(features, sample_index, feature_index) - self.means[feature_index])
                        / self.scales[feature_index],
                );
            }
        }

        Ok(NDArray::from_shape_vec([sample_count, feature_count], transformed)?)
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
    let offset = features.offset()
        + sample_index * features.strides()[0]
        + feature_index * features.strides()[1];

    features.data()[offset]
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::StandardScaler;
    use crate::AtlasMlError;

    #[test]
    fn fits_and_transforms_ordinary_features() {
        let features =
            NDArray::from_shape_vec([3, 2], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let scaler = StandardScaler::fit(&features).unwrap();
        let transformed = scaler.transform(&features).unwrap();
        let scale = (8.0_f64 / 3.0).sqrt();

        assert_eq!(scaler.means(), &[3.0, 4.0]);
        assert_eq!(scaler.scales(), &[scale, scale]);
        assert_close(
            transformed.data(),
            &[-2.0 / scale, -2.0 / scale, 0.0, 0.0, 2.0 / scale, 2.0 / scale],
        );
    }

    #[test]
    fn transforms_constant_features_to_zero() {
        let features = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 5.0, 3.0, 5.0]).unwrap();
        let scaler = StandardScaler::fit(&features).unwrap();

        assert_eq!(scaler.scales(), &[1.0, 1.0]);
        assert_eq!(scaler.transform(&features).unwrap().data(), &[-1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn supports_logical_feature_views() {
        let features =
            NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let view = features.view().transpose();
        let scaler = StandardScaler::fit(&view).unwrap();
        let scale = (2.0_f64 / 3.0).sqrt();

        assert_eq!(scaler.means(), &[2.0, 5.0]);
        assert_close(
            scaler.transform(&view).unwrap().data(),
            &[-1.0 / scale, -1.0 / scale, 0.0, 0.0, 1.0 / scale, 1.0 / scale],
        );
    }

    #[test]
    fn transforms_empty_feature_batches() {
        let scaler = StandardScaler::fit(
            &NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap(),
        )
        .unwrap();
        let transformed = scaler.transform(&NDArray::<f64>::zeros([0, 2]).unwrap()).unwrap();

        assert_eq!(transformed.shape(), &[0, 2]);
        assert!(transformed.data().is_empty());
    }

    #[test]
    fn rejects_non_finite_features() {
        let finite = NDArray::from_shape_vec([1, 1], vec![1.0_f64]).unwrap();
        let non_finite = NDArray::from_shape_vec([1, 1], vec![f64::NAN]).unwrap();
        let scaler = StandardScaler::fit(&finite).unwrap();

        assert_eq!(
            StandardScaler::fit(&non_finite),
            Err(AtlasMlError::NonFiniteInput { op: "standard_scaler_fit" })
        );
        assert_eq!(
            scaler.transform(&non_finite).map(|_| ()),
            Err(AtlasMlError::NonFiniteInput { op: "standard_scaler_transform" })
        );
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
