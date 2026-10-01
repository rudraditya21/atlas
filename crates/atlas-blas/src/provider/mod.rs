#[cfg(atlas_blas_accelerate)]
mod accelerate;
#[cfg(atlas_blas_blis)]
mod blis;
#[cfg(atlas_blas_mkl)]
mod mkl;
#[cfg(atlas_blas_openblas)]
mod openblas;

#[cfg(atlas_blas_accelerate)]
pub(crate) use accelerate::*;
#[cfg(atlas_blas_blis)]
pub(crate) use blis::*;
#[cfg(atlas_blas_mkl)]
pub(crate) use mkl::*;
#[cfg(atlas_blas_openblas)]
pub(crate) use openblas::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provider {
    Accelerate,
    Blis,
    OneMkl,
    OpenBlas,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Threading {
    ProviderDefault,
    SingleThreaded,
}

impl Provider {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Accelerate => "accelerate",
            Self::Blis => "blis",
            Self::OneMkl => "onemkl",
            Self::OpenBlas => "openblas",
        }
    }

    pub const fn thread_control_available(self) -> bool {
        true
    }
}

#[cfg(atlas_blas_accelerate)]
const ACTIVE_PROVIDER: Option<Provider> = Some(Provider::Accelerate);
#[cfg(atlas_blas_openblas)]
const ACTIVE_PROVIDER: Option<Provider> = Some(Provider::OpenBlas);
#[cfg(atlas_blas_blis)]
const ACTIVE_PROVIDER: Option<Provider> = Some(Provider::Blis);
#[cfg(atlas_blas_mkl)]
const ACTIVE_PROVIDER: Option<Provider> = Some(Provider::OneMkl);
#[cfg(not(any(atlas_blas_accelerate, atlas_blas_blis, atlas_blas_mkl, atlas_blas_openblas)))]
const ACTIVE_PROVIDER: Option<Provider> = None;

pub const fn active_provider() -> Option<Provider> {
    ACTIVE_PROVIDER
}

#[cfg(not(any(atlas_blas_accelerate, atlas_blas_blis, atlas_blas_mkl, atlas_blas_openblas)))]
pub(crate) fn with_threading<R>(_: Threading, operation: impl FnOnce() -> R) -> R {
    operation()
}
