pub(crate) mod dense;
pub(crate) mod factorization;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;
pub(crate) mod simd;
