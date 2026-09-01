pub(crate) mod broadcast;
pub(crate) mod concat;
pub(crate) mod flatten;
pub(crate) mod permutation;
pub(crate) mod ravel;
pub(crate) mod repeat;
pub(crate) mod reshape;
pub(crate) mod shape_ops;
pub(crate) mod split;
pub(crate) mod stack;
pub(crate) mod stride;
pub(crate) mod transpose;

pub(crate) use crate::internal::shape::element_count;
