import atlas


def test_package_imports_its_native_module() -> None:
    assert atlas._native.version() == atlas.__version__


def test_package_version_matches_native_version() -> None:
    assert isinstance(atlas.__version__, str)
    assert atlas.__version__
