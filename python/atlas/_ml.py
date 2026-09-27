"""Machine-learning bindings exposed by Atlas."""

from . import _native
from ._support import _coerce_arrays

ConfusionMatrix = _native.ConfusionMatrix
ClassificationReport = _native.ClassificationReport
TrainTestSplit = _native.TrainTestSplit
KFold = _native.KFold
LinearRegression = _native.LinearRegression
RidgeRegression = _native.RidgeRegression
BinaryLogisticRegression = _native.BinaryLogisticRegression
BinaryPerceptron = _native.BinaryPerceptron
NearestCentroidClassifier = _native.NearestCentroidClassifier
GaussianNaiveBayes = _native.GaussianNaiveBayes
KnnClassifier = _native.KnnClassifier
KnnRegressor = _native.KnnRegressor
DecisionStumpClassifier = _native.DecisionStumpClassifier
BinaryGiniSplit = _native.BinaryGiniSplit
StandardScaler = _native.StandardScaler
MinMaxScaler = _native.MinMaxScaler
evaluate_binary_gini_split = _native.evaluate_binary_gini_split
accuracy = _native.accuracy
log_loss = _native.log_loss
mean_absolute_error = _native.mean_absolute_error
mean_squared_error = _native.mean_squared_error
r_squared = _native.r_squared
confusion_matrix = _native.confusion_matrix
classification_report = _native.classification_report
train_test_split = _native.train_test_split
k_fold_split = _native.k_fold_split
stratified_k_fold_split = _native.stratified_k_fold_split

evaluate_binary_gini_split = _coerce_arrays(
    evaluate_binary_gini_split, required=((0, "feature_values"), (1, "labels"))
)
for _name in (
    "accuracy",
    "mean_absolute_error",
    "mean_squared_error",
    "r_squared",
    "confusion_matrix",
    "classification_report",
):
    globals()[_name] = _coerce_arrays(
        globals()[_name], required=((0, "actual"), (1, "predicted"))
    )
log_loss = _coerce_arrays(log_loss, required=((0, "actual"), (1, "probabilities")))
train_test_split = _coerce_arrays(
    train_test_split, required=((0, "features"), (1, "targets"))
)
k_fold_split = _coerce_arrays(k_fold_split, required=((0, "features"),))
stratified_k_fold_split = _coerce_arrays(
    stratified_k_fold_split, required=((0, "features"), (1, "labels"))
)
