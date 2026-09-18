import atlas


def test_atlas_exceptions_have_the_expected_inheritance() -> None:
    assert issubclass(atlas.ShapeError, atlas.AtlasError)
    assert issubclass(atlas.AxisError, atlas.ShapeError)
    assert issubclass(atlas.SliceError, atlas.ShapeError)
    assert issubclass(atlas.NumericError, atlas.AtlasError)
    assert issubclass(atlas.ModelError, atlas.AtlasError)


def test_atlas_exceptions_are_directly_constructible() -> None:
    for error_type in [
        atlas.AtlasError,
        atlas.ShapeError,
        atlas.AxisError,
        atlas.SliceError,
        atlas.NumericError,
        atlas.ModelError,
    ]:
        error = error_type("invalid input")

        assert error.__class__.__name__ == error_type.__name__
        assert str(error) == "invalid input"
        assert isinstance(error, atlas.AtlasError)
