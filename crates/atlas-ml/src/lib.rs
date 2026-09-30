//! Machine-learning algorithms built on Atlas arrays.

mod core;
mod decision_stump;
mod decision_tree;
mod error;
mod gaussian_naive_bayes;
mod internal;
mod knn;
mod linear_regression;
mod logistic_regression;
mod metrics;
mod nearest_centroid;
mod perceptron;
mod preprocessing;
mod ridge_regression;

pub use core::{
    k_fold::{KFold, k_fold_split, stratified_k_fold_split},
    train_test_split::{
        TrainTestSplit, model_evaluation_split, stratified_train_test_split, train_test_split,
    },
};

pub use decision_stump::DecisionStumpClassifier;
pub use decision_tree::{BinaryGiniSplit, evaluate_binary_gini_split};
pub use error::{AtlasMlError, AtlasMlResult};
pub use gaussian_naive_bayes::{GaussianNaiveBayes, GaussianNaiveBayesConfig};
pub use knn::{
    classifier::KnnClassifier,
    config::{AUTO_BRUTE_FORCE_MAX_SAMPLES, KnnConfig, KnnSearchAlgorithm, KnnWeighting},
    regressor::KnnRegressor,
};
pub use linear_regression::LinearRegression;
pub use logistic_regression::{BinaryLogisticRegression, LogisticRegressionConfig};
pub use metrics::{
    ClassificationReport, ConfusionMatrix, binary_log_loss, classification_accuracy,
    classification_report, coefficient_of_determination, confusion_matrix, mean_absolute_error,
    mean_squared_error,
};
pub use nearest_centroid::NearestCentroidClassifier;
pub use perceptron::{BinaryPerceptron, PerceptronConfig, PerceptronShufflePolicy};
pub use preprocessing::{LabelEncoder, MinMaxScaler, StandardScaler};
pub use ridge_regression::{RidgeRegression, RidgeRegressionConfig};
