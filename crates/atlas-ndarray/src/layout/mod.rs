pub(crate) mod broadcast;
pub(crate) mod reshape;
pub(crate) mod stride;
pub(crate) mod transpose;

pub(crate) use crate::internal::shape::{compute_strides, element_count};
