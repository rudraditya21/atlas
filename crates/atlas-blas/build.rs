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
            println!("cargo:rustc-link-lib=static=openblas");
            enable_provider("atlas_blas_openblas");
        }
        Some(Selection::BlisSystem) => {
            println!("cargo:rustc-link-lib=blis");
            enable_provider("atlas_blas_blis");
        }
        Some(Selection::MklSystem) => {
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
