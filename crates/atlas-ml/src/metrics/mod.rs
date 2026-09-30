mod classification;
mod confusion_matrix;
mod regression;

pub use classification::{
    ClassificationReport, binary_log_loss, classification_accuracy, classification_report,
};
pub use confusion_matrix::{ConfusionMatrix, confusion_matrix};
pub use regression::{coefficient_of_determination, mean_absolute_error, mean_squared_error};
