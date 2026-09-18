use pyo3::{Python, marker::Ungil};

pub(crate) fn without_gil<T, F>(py: Python<'_>, computation: F) -> T
where
    F: Ungil + FnOnce() -> T,
    T: Ungil,
{
    py.detach(computation)
}
