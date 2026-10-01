use std::env;

#[derive(Clone, Copy)]
enum Selection {
    Accelerate,
    BlisSystem,
    MklSystem,
    Native,
    OpenBlasStatic,
    OpenBlasSystem,
}

fn main() {
    for configuration in [
        "atlas_blas",
        "atlas_blas_accelerate",
        "atlas_blas_blis",
        "atlas_blas_mkl",
        "atlas_blas_openblas",
    ] {
        println!("cargo:rustc-check-cfg=cfg({configuration})");
    }
    for feature in
        ["ACCELERATE", "BLIS_SYSTEM", "MKL_SYSTEM", "NATIVE", "OPENBLAS_STATIC", "OPENBLAS_SYSTEM"]
    {
        println!("cargo:rerun-if-env-changed=CARGO_FEATURE_{feature}");
    }
    println!("cargo:rerun-if-env-changed=ATLAS_OPENBLAS_LIBRARY_DIR");
    println!("cargo:rerun-if-env-changed=ATLAS_MKL_LIBRARY_DIR");
    println!("cargo:rerun-if-env-changed=OPENBLAS_DYNAMIC_ARCH");

    let target = env::var("TARGET").unwrap_or_default();
    let is_macos = target.contains("apple-darwin");
    let mut selected = Vec::new();
    select_feature(&mut selected, "ACCELERATE", Selection::Accelerate);
    select_feature(&mut selected, "BLIS_SYSTEM", Selection::BlisSystem);
    select_feature(&mut selected, "MKL_SYSTEM", Selection::MklSystem);
    select_feature(&mut selected, "NATIVE", Selection::Native);
    select_feature(&mut selected, "OPENBLAS_STATIC", Selection::OpenBlasStatic);
    select_feature(&mut selected, "OPENBLAS_SYSTEM", Selection::OpenBlasSystem);

    if selected.len() > 1 {
        panic!("atlas-blas provider features are mutually exclusive");
    }

    let selection = selected.first().copied().or(is_macos.then_some(Selection::Accelerate));
    match selection {
        Some(Selection::Accelerate) => {
            if !is_macos {
                panic!("the accelerate provider is supported only on macOS");
            }
            println!("cargo:rustc-link-lib=framework=Accelerate");
            enable_provider("atlas_blas_accelerate");
        }
        Some(Selection::OpenBlasSystem) => {
            println!("cargo:rustc-link-lib=openblas");
            enable_provider("atlas_blas_openblas");
        }
        Some(Selection::OpenBlasStatic) => {
            if env::var("OPENBLAS_DYNAMIC_ARCH").unwrap_or_default() != "1" {
                panic!("openblas-static requires an OpenBLAS library built with DYNAMIC_ARCH=1");
            }
            if let Some(directory) = env::var_os("ATLAS_OPENBLAS_LIBRARY_DIR") {
                println!("cargo:rustc-link-search=native={}", directory.to_string_lossy());
            }
            println!("cargo:rustc-link-lib=static=openblas");
            enable_provider("atlas_blas_openblas");
        }
        Some(Selection::BlisSystem) => {
            println!("cargo:rustc-link-lib=blis");
            enable_provider("atlas_blas_blis");
        }
        Some(Selection::MklSystem) => {
            if let Some(directory) = env::var_os("ATLAS_MKL_LIBRARY_DIR") {
                println!("cargo:rustc-link-search=native={}", directory.to_string_lossy());
            }
            println!("cargo:rustc-link-lib=mkl_rt");
            enable_provider("atlas_blas_mkl");
        }
        Some(Selection::Native) | None => {}
    }
}

fn select_feature(selected: &mut Vec<Selection>, feature: &str, selection: Selection) {
    if env::var_os(format!("CARGO_FEATURE_{feature}")).is_some() {
        selected.push(selection);
    }
}

fn enable_provider(configuration: &str) {
    println!("cargo:rustc-cfg=atlas_blas");
    println!("cargo:rustc-cfg={configuration}");
}
