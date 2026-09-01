pub(crate) mod indexing;
pub(crate) mod iter;
pub(crate) mod slicing;
// This module contains the ArrayView type; retaining the name keeps the public module layout clear.
#[allow(clippy::module_inception)]
pub(crate) mod view;

pub(crate) use view::ArrayView;
