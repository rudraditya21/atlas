import atlas
import atlas.errors


def test_public_error_exports_are_complete_and_consistent() -> None:
    assert tuple(atlas.errors.__all__) == (
        "AtlasError",
        "AxisError",
        "ModelError",
        "NumericError",
        "ShapeError",
        "SliceError",
    )

    for name in atlas.errors.__all__:
        assert name in atlas.__all__
        assert getattr(atlas, name) is getattr(atlas.errors, name)


def test_public_error_classes_use_root_package_ownership() -> None:
    for name in atlas.errors.__all__:
        error_type = getattr(atlas.errors, name)
        assert issubclass(error_type, Exception)
        assert error_type.__module__ == "atlas"
