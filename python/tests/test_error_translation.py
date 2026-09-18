import pytest

import atlas


@pytest.mark.parametrize(
    ("function_name", "error_type", "message"),
    [
        ("_raise_ndarray_error", atlas.AxisError, "invalid axis"),
        ("_raise_stats_error", atlas.NumericError, "invalid quantile"),
        ("_raise_linalg_error", atlas.ShapeError, "shape mismatch"),
        ("_raise_random_error", atlas.NumericError, "invalid argument"),
        ("_raise_ml_error", atlas.ModelError, "invalid argument"),
    ],
)
def test_native_errors_map_to_public_python_exceptions(
    function_name: str,
    error_type: type[atlas.AtlasError],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        getattr(atlas._native, function_name)()
