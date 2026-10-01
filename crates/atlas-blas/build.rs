use std::env;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(atlas_blas)");
    println!("cargo:rustc-check-cfg=cfg(atlas_blas_accelerate)");
    println!("cargo:rustc-check-cfg=cfg(atlas_blas_openblas)");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_OPENBLAS");

    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("apple-darwin") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-cfg=atlas_blas");
        println!("cargo:rustc-cfg=atlas_blas_accelerate");
    } else if env::var_os("CARGO_FEATURE_OPENBLAS").is_some() {
        println!("cargo:rustc-link-lib=openblas");
        println!("cargo:rustc-cfg=atlas_blas");
        println!("cargo:rustc-cfg=atlas_blas_openblas");
    }
}
