import numpy as np

import atlas


def test_generator_helpers_are_seeded_and_context_free() -> None:
    first = atlas.default_rng(17)
    second = atlas.seeded_rng(17)

    np.testing.assert_array_equal(first.random((2, 3)), second.random((2, 3)))


def test_generator_state_restores_the_next_samples() -> None:
    generator = atlas.default_rng(23)

    generator.normal(4)
    state = generator.get_state()
    expected = generator.randint(6, 0, 10)
    generator.random(5)
    generator.set_state(state)

    np.testing.assert_array_equal(generator.randint(6, 0, 10), expected)


def test_generator_state_can_seed_an_independent_generator() -> None:
    source = atlas.default_rng(31)
    state = source.get_state()
    restored = atlas.default_rng(0)
    restored.set_state(state)

    np.testing.assert_array_equal(source.bernoulli(8), restored.bernoulli(8))
