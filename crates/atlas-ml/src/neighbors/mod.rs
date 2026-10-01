mod knn;
mod nearest_centroid;

pub use knn::{
    AUTO_BRUTE_FORCE_MAX_SAMPLES, KnnClassifier, KnnConfig, KnnRegressor, KnnSearchAlgorithm,
    KnnWeighting,
};
pub use nearest_centroid::NearestCentroidClassifier;
