mod backend;
mod ball_tree;
mod brute_force;
mod classifier;
mod config;
mod index;
mod kd_tree;
mod metric;
mod regressor;
mod top_k;

pub use classifier::KnnClassifier;
pub use config::{AUTO_BRUTE_FORCE_MAX_SAMPLES, KnnConfig, KnnSearchAlgorithm, KnnWeighting};
pub use regressor::KnnRegressor;
