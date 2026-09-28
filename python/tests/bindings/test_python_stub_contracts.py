import ast
from pathlib import Path

import atlas
from atlas import _exports


PACKAGE_DIRECTORY = Path(atlas.__file__).parent


def _stub_exports(module: str) -> set[str]:
    tree = ast.parse((PACKAGE_DIRECTORY / f"{module}.pyi").read_text())
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        and not node.name.startswith("_")
    }


def test_public_submodule_stubs_match_runtime_exports() -> None:
    for module in (
        atlas.linalg,
        atlas.statistics,
        atlas.random,
        atlas.ml,
        atlas.interop,
    ):
        declared = _stub_exports(module.__name__.rsplit(".", maxsplit=1)[-1])
        assert set(module.__all__) <= declared


def test_root_stub_covers_array_exports_and_numpy_aliases() -> None:
    declared = _stub_exports("__init__")

    assert set(_exports.ARRAY_EXPORTS) <= declared
    assert set(_exports.NUMPY_COMPATIBILITY_ALIASES) <= declared


def test_package_includes_the_pep_561_marker() -> None:
    assert (PACKAGE_DIRECTORY / "py.typed").is_file()
