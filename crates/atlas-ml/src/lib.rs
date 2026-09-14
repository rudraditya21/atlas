//! Machine-learning algorithms built on Atlas arrays.

mod core;
mod gaussian_naive_bayes;
mod knn;
mod linear_regression;
mod logistic_regression;
mod nearest_centroid;
mod perceptron;
mod ridge_regression;

pub use core::{
    confusion_matrix::{ConfusionMatrix, confusion_matrix},
    error::{AtlasMlError, AtlasMlResult},
    k_fold::{KFold, k_fold_split, stratified_k_fold_split},
    label_encoder::LabelEncoder,
    metrics::{
        ClassificationReport, binary_log_loss, classification_accuracy, classification_report,
        coefficient_of_determination, mean_absolute_error, mean_squared_error,
    },
    min_max_scaler::MinMaxScaler,
    standard_scaler::StandardScaler,
    train_test_split::{
        TrainTestSplit, model_evaluation_split, stratified_train_test_split, train_test_split,
    },
};

pub use gaussian_naive_bayes::{GaussianNaiveBayes, GaussianNaiveBayesConfig};
pub use knn::{
    classifier::KnnClassifier,
    config::{AUTO_BRUTE_FORCE_MAX_SAMPLES, KnnConfig, KnnSearchAlgorithm, KnnWeighting},
    regressor::KnnRegressor,
};
pub use linear_regression::LinearRegression;
pub use logistic_regression::{BinaryLogisticRegression, LogisticRegressionConfig};
pub use nearest_centroid::NearestCentroidClassifier;
pub use perceptron::{PerceptronConfig, PerceptronShufflePolicy};
pub use ridge_regression::{RidgeRegression, RidgeRegressionConfig};
