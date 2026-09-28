"""Random-number generation bindings exposed by Atlas."""

from . import _native

Generator = _native.Generator


def default_rng(seed=0):
    """Return a deterministic generator independent of module-global state."""

    return Generator(seed)


def seeded_rng(seed):
    """Return a generator initialized from ``seed``."""

    return default_rng(seed)
