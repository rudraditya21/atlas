"""Statistical functions exposed by Atlas."""

from . import _native
from ._support import _coerce_arrays, _coerce_binary_operands, _coerce_weighted_operands

kurtosis = _native.kurtosis
skewness = _native.skewness
median = _native.median
quantile = _native.quantile
median_axis = _native.median_axis
quantile_axis = _native.quantile_axis
weighted_mean = _native.weighted_mean
weighted_variance = _native.weighted_variance
covariance = _native.covariance
correlation = _native.correlation
covariance_matrix = _native.covariance_matrix
correlation_matrix = _native.correlation_matrix


def variance(value, *, ddof=0):
    return _native.variance(value, ddof)


def stddev(value, *, ddof=0):
    return _native.stddev(value, ddof)


def quantile(value, q, *, interpolation="linear"):
    return _native.quantile(value, q, interpolation)


def quantile_axis(value, q, axis, *, interpolation="linear"):
    return _native.quantile_axis(value, q, axis, interpolation)


for _name in (
    "variance",
    "stddev",
    "kurtosis",
    "skewness",
    "median",
    "quantile",
    "median_axis",
    "quantile_axis",
    "covariance_matrix",
    "correlation_matrix",
):
    globals()[_name] = _coerce_arrays(globals()[_name], required=((0, "value"),))

for _name in ("covariance", "correlation"):
    globals()[_name] = _coerce_binary_operands(globals()[_name])

for _name in ("weighted_mean", "weighted_variance"):
    globals()[_name] = _coerce_weighted_operands(globals()[_name])
