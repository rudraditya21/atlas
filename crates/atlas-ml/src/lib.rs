//! Machine-learning algorithms built on Atlas arrays.

mod core;
mod knn;

pub use core::error::{AtlasMlError, AtlasMlResult};

pub use knn::config::{KnnConfig, KnnSearchAlgorithm};
