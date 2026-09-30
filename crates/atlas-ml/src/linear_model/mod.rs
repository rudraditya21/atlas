mod linear_regression;
mod logistic_regression;
mod perceptron;
mod ridge_regression;

pub use linear_regression::LinearRegression;
pub use logistic_regression::{BinaryLogisticRegression, LogisticRegressionConfig};
pub use perceptron::{BinaryPerceptron, PerceptronConfig, PerceptronShufflePolicy};
pub use ridge_regression::{RidgeRegression, RidgeRegressionConfig};
