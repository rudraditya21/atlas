from inspect import Parameter, signature

import atlas
import atlas.interop
import atlas.linalg
import atlas.ml
import atlas.random
import atlas.statistics


def test_public_submodules_export_their_declared_api() -> None:
    for module in (
        atlas.linalg,
        atlas.statistics,
        atlas.random,
        atlas.ml,
        atlas.interop,
    ):
        assert module.__all__
        assert all(hasattr(module, name) for name in module.__all__)


def test_package_root_reexports_public_domain_members() -> None:
    for module in (
        atlas.linalg,
        atlas.statistics,
        atlas.random,
        atlas.ml,
        atlas.interop,
    ):
        for name in module.__all__:
            root_member = getattr(atlas, name)
            module_member = getattr(module, name)
            assert root_member.__name__ == module_member.__name__
            assert callable(root_member) or isinstance(root_member, type)


def test_public_signatures_preserve_keyword_only_configuration() -> None:
    expectations = (
        (atlas.variance, "ddof"),
        (atlas.stddev, "ddof"),
        (atlas.quantile, "interpolation"),
        (atlas.quantile_axis, "interpolation"),
        (atlas.train_test_split, "test_ratio"),
        (atlas.train_test_split, "seed"),
        (atlas.k_fold_split, "seed"),
        (atlas.stratified_k_fold_split, "seed"),
    )

    for function, parameter in expectations:
        assert signature(function).parameters[parameter].kind is Parameter.KEYWORD_ONLY
