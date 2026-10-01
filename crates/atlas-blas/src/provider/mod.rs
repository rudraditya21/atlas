#[cfg(atlas_blas_accelerate)]
mod accelerate;
#[cfg(atlas_blas_openblas)]
mod linked;

#[cfg(atlas_blas_accelerate)]
pub(crate) use accelerate::*;
#[cfg(atlas_blas_openblas)]
pub(crate) use linked::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    Accelerate,
    OpenBlas,
}

#[cfg(atlas_blas_accelerate)]
const ACTIVE_PROVIDER: Option<Provider> = Some(Provider::Accelerate);
#[cfg(atlas_blas_openblas)]
const ACTIVE_PROVIDER: Option<Provider> = Some(Provider::OpenBlas);
#[cfg(not(any(atlas_blas_accelerate, atlas_blas_openblas)))]
const ACTIVE_PROVIDER: Option<Provider> = None;

pub const fn active_provider() -> Option<Provider> {
    ACTIVE_PROVIDER
}
