pub(crate) mod knn;
mod nearest_centroid;

pub use knn::{
    classifier::KnnClassifier,
    config::{AUTO_BRUTE_FORCE_MAX_SAMPLES, KnnConfig, KnnSearchAlgorithm, KnnWeighting},
    regressor::KnnRegressor,
};
pub use nearest_centroid::NearestCentroidClassifier;
