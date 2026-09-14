//! Machine-learning algorithms built on Atlas arrays.

mod core;
mod knn;

pub use core::{
    error::{AtlasMlError, AtlasMlResult},
    label_encoder::LabelEncoder,
};

pub use knn::{
    classifier::KnnClassifier,
    config::{AUTO_BRUTE_FORCE_MAX_SAMPLES, KnnConfig, KnnSearchAlgorithm, KnnWeighting},
    regressor::KnnRegressor,
};
