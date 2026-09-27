//! Shared PyO3 registration helpers.

#[macro_export]
macro_rules! register_functions {
    ($module:expr; $($function:ident),+ $(,)?) => {
        $(
            $module.add_function(pyo3::wrap_pyfunction!($function, $module)?)?;
        )+
    };
}
