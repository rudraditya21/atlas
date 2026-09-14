//! Machine-learning algorithms built on Atlas arrays.

mod core;
mod knn;
mod linear_regression;
mod nearest_centroid;

pub use core::{
    confusion_matrix::{ConfusionMatrix, confusion_matrix},
    error::{AtlasMlError, AtlasMlResult},
    k_fold::{KFold, k_fold_split},
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
pub use linear_regression::LinearRegression;
pub use nearest_centroid::NearestCentroidClassifier;
