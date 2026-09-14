//! Machine-learning algorithms built on Atlas arrays.

mod core;
mod knn;
mod nearest_centroid;

pub use core::{
    error::{AtlasMlError, AtlasMlResult},
    label_encoder::LabelEncoder,
    metrics::{classification_accuracy, mean_absolute_error, mean_squared_error},
    min_max_scaler::MinMaxScaler,
    standard_scaler::StandardScaler,
    train_test_split::{TrainTestSplit, train_test_split},
};

pub use knn::{
    classifier::KnnClassifier,
    config::{AUTO_BRUTE_FORCE_MAX_SAMPLES, KnnConfig, KnnSearchAlgorithm, KnnWeighting},
    regressor::KnnRegressor,
};
pub use nearest_centroid::NearestCentroidClassifier;
