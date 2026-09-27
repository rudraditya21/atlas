"""Public Python API for Atlas."""

from functools import wraps

import numpy as np

from . import _arrays, _interop, _linalg, _ml, _native, _random, _statistics
from .errors import (
    AtlasError,
    AxisError,
    ModelError,
    NumericError,
    ShapeError,
    SliceError,
)

__version__ = _native.version()


def _with_keepdims(function):
    def wrapper(value, axis=None, *, keepdims=False):
        if keepdims and axis is None:
            raise ValueError("keepdims requires a single axis")
        result = function(value, axis=axis)
        return np.expand_dims(result, axis) if keepdims else result

    wrapper.__name__ = function.__name__
    wrapper.__doc__ = function.__doc__
    return wrapper


def _publish(function):
    """Expose a native callable as a public Atlas function."""

    @wraps(function)
    def wrapper(*args, **kwargs):
        return function(*args, **kwargs)

    wrapper.__module__ = __name__
    return wrapper


__all__ = sorted(
    [
        "AtlasError",
        "AxisError",
        "BinaryGiniSplit",
        "BinaryLogisticRegression",
        "BinaryPerceptron",
        "ClassificationReport",
        "ConjugateGradientResult",
        "ConfusionMatrix",
        "DecisionStumpClassifier",
        "Generator",
        "GaussianNaiveBayes",
        "KFold",
        "KnnClassifier",
        "KnnRegressor",
        "LinearRegression",
        "MinMaxScaler",
        "ModelError",
        "NearestCentroidClassifier",
        "NumericError",
        "RidgeRegression",
        "ShapeError",
        "SliceError",
        "StandardScaler",
        "TrainTestSplit",
        "__version__",
        "abs",
        "add",
        "accuracy",
        "all",
        "all_axis",
        "allclose",
        "any",
        "any_axis",
        "arange",
        "argmax",
        "argmax_axis",
        "argmin",
        "argmin_axis",
        "argpartition",
        "argsort",
        "argwhere",
        "asarray",
        "ascontiguousarray",
        "astype",
        "bitwise_and",
        "bitwise_not",
        "bitwise_or",
        "bitwise_xor",
        "can_cast",
        "cholesky",
        "choose",
        "classification_report",
        "clip",
        "common_type",
        "concatenate",
        "conjugate_gradient",
        "confusion_matrix",
        "copy",
        "copyto",
        "correlation",
        "correlation_matrix",
        "count_true",
        "covariance",
        "covariance_matrix",
        "cumprod",
        "cumprod_axis",
        "cumsum",
        "cumsum_axis",
        "det",
        "diag",
        "divide",
        "dot",
        "dtype",
        "equal",
        "evaluate_binary_gini_split",
        "expand_dims",
        "eye",
        "finfo",
        "flatten",
        "flip",
        "from_arrow_primitive",
        "from_arrow_record_batch",
        "full",
        "full_like",
        "greater",
        "greater_equal",
        "identity",
        "iinfo",
        "inverse",
        "isfinite",
        "isinf",
        "isnan",
        "issubdtype",
        "k_fold_split",
        "kurtosis",
        "least_squares",
        "less",
        "less_equal",
        "linspace",
        "log_loss",
        "masked_fill",
        "matmul",
        "matrix_norm",
        "matrix_rank",
        "max",
        "max_axis",
        "mean",
        "mean_absolute_error",
        "mean_axis",
        "mean_squared_error",
        "median",
        "median_axis",
        "min",
        "min_axis",
        "min_scalar_type",
        "moveaxis",
        "multiply",
        "nanmax",
        "nanmean",
        "nanmin",
        "nanstd",
        "ndim",
        "neg",
        "nonzero",
        "norm",
        "not_equal",
        "ones",
        "ones_like",
        "pad",
        "partition",
        "promote_types",
        "quantile",
        "quantile_axis",
        "qr",
        "r_squared",
        "ravel",
        "repeat",
        "reshape",
        "result_type",
        "roll",
        "round",
        "searchsorted",
        "select",
        "shape",
        "shares_memory",
        "sign",
        "size",
        "skewness",
        "slogdet",
        "solve",
        "solve_spd",
        "solve_transpose",
        "sort",
        "split",
        "squeeze",
        "stack",
        "std",
        "stddev",
        "stratified_k_fold_split",
        "subtract",
        "sum",
        "sum_axis",
        "swap_axes",
        "swapaxes",
        "symmetric_eigendecomposition",
        "take",
        "tile",
        "to_arrow_primitive",
        "to_arrow_record_batch",
        "trace",
        "train_test_split",
        "transpose",
        "unique",
        "var",
        "variance",
        "weighted_mean",
        "weighted_variance",
        "where",
        "zeros",
        "zeros_like",
    ]
)

for _module in (_arrays, _interop, _linalg, _ml, _random, _statistics):
    for _name in __all__:
        if hasattr(_module, _name):
            globals()[_name] = getattr(_module, _name)

for _name in (
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "broadcast_arrays",
    "broadcast_to",
    "empty_like",
    "may_share_memory",
    "put",
):
    _value = getattr(_arrays, _name)
    _value.__module__ = __name__
    globals()[_name] = _value

for _name in ("all", "any", "sum", "mean", "min", "max", "argmin", "argmax"):
    globals()[_name] = _with_keepdims(globals()[_name])

for _name in __all__:
    if _name in {"std", "swapaxes", "var"}:
        continue
    _value = globals()[_name]
    if isinstance(_value, type):
        _value.__module__ = __name__
    elif callable(_value):
        if getattr(_value, "__module__", None) == _native.__name__:
            _value = _publish(_value)
            globals()[_name] = _value
        else:
            _value.__module__ = __name__

swapaxes = swap_axes
var = variance
std = stddev
