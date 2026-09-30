//! Machine-learning algorithms built on Atlas arrays.

mod error;
mod internal;
mod knn;
mod linear_model;
mod metrics;
mod model_selection;
mod naive_bayes;
mod nearest_centroid;
mod preprocessing;
mod tree;

pub use error::{AtlasMlError, AtlasMlResult};
pub use knn::{
    classifier::KnnClassifier,
    config::{AUTO_BRUTE_FORCE_MAX_SAMPLES, KnnConfig, KnnSearchAlgorithm, KnnWeighting},
    regressor::KnnRegressor,
};
pub use linear_model::{
    BinaryLogisticRegression, BinaryPerceptron, LinearRegression, LogisticRegressionConfig,
    PerceptronConfig, PerceptronShufflePolicy, RidgeRegression, RidgeRegressionConfig,
};
pub use metrics::{
    ClassificationReport, ConfusionMatrix, binary_log_loss, classification_accuracy,
    classification_report, coefficient_of_determination, confusion_matrix, mean_absolute_error,
    mean_squared_error,
};
pub use model_selection::{
    KFold, TrainTestSplit, k_fold_split, model_evaluation_split, stratified_k_fold_split,
    stratified_train_test_split, train_test_split,
};
pub use naive_bayes::{GaussianNaiveBayes, GaussianNaiveBayesConfig};
pub use nearest_centroid::NearestCentroidClassifier;
pub use preprocessing::{LabelEncoder, MinMaxScaler, StandardScaler};
pub use tree::{BinaryGiniSplit, DecisionStumpClassifier, evaluate_binary_gini_split};
