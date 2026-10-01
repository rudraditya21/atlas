#[cfg(atlas_blas_accelerate)]
mod accelerate;
#[cfg(atlas_blas_blis)]
mod blis;
#[cfg(atlas_blas_mkl)]
mod linked;
#[cfg(atlas_blas_openblas)]
mod openblas;

#[cfg(atlas_blas_accelerate)]
pub(crate) use accelerate::*;
#[cfg(atlas_blas_blis)]
pub(crate) use blis::*;
#[cfg(atlas_blas_mkl)]
pub(crate) use linked::*;
#[cfg(atlas_blas_openblas)]
pub(crate) use openblas::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    Accelerate,
    Blis,
    Mkl,
    OpenBlas,
}

#[cfg(atlas_blas_accelerate)]
const ACTIVE_PROVIDER: Option<Provider> = Some(Provider::Accelerate);
#[cfg(atlas_blas_openblas)]
const ACTIVE_PROVIDER: Option<Provider> = Some(Provider::OpenBlas);
#[cfg(atlas_blas_blis)]
const ACTIVE_PROVIDER: Option<Provider> = Some(Provider::Blis);
#[cfg(atlas_blas_mkl)]
const ACTIVE_PROVIDER: Option<Provider> = Some(Provider::Mkl);
#[cfg(not(any(atlas_blas_accelerate, atlas_blas_blis, atlas_blas_mkl, atlas_blas_openblas)))]
const ACTIVE_PROVIDER: Option<Provider> = None;

pub const fn active_provider() -> Option<Provider> {
    ACTIVE_PROVIDER
}
