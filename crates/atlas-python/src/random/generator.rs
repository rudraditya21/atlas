use atlas_random::AtlasRng;
use pyo3::prelude::*;

#[pyclass(module = "atlas._native")]
pub(crate) struct Generator {
    #[allow(dead_code, reason = "random distribution methods are registered incrementally")]
    pub(crate) rng: AtlasRng,
}

#[pymethods]
impl Generator {
    #[new]
    fn new(seed: u64) -> Self {
        Self { rng: AtlasRng::seed_from_u64(seed) }
    }
}
