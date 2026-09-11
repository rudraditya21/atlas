//! Machine-learning algorithms built on Atlas arrays.

mod core;
mod knn;

pub use core::error::{AtlasMlError, AtlasMlResult};

pub use knn::{
    classifier::KnnClassifier,
    config::{KnnConfig, KnnSearchAlgorithm},
    regressor::KnnRegressor,
};
