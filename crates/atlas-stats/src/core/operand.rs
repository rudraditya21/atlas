use std::slice::Iter;

use atlas_ndarray::{ArrayView, ArrayViewIter, NDArray, Numeric, checked_element_count};

use crate::core::error::AtlasStatsResult;

#[derive(Clone, Debug)]
pub enum StatsOperand<'a, T: Numeric> {
    Array(&'a NDArray<T>),
    View(ArrayView<'a, T>),
}

pub(crate) enum StatsOperandIter<'a, T: Numeric> {
    Array(Iter<'a, T>),
    View(ArrayViewIter<'a, T>),
}

impl<'a, T: Numeric> Iterator for StatsOperandIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Array(iter) => iter.next(),
            Self::View(iter) => iter.next(),
        }
    }
}

impl<'a, T: Numeric> From<&'a NDArray<T>> for StatsOperand<'a, T> {
    fn from(value: &'a NDArray<T>) -> Self {
        Self::Array(value)
    }
}

impl<'a, T: Numeric> From<ArrayView<'a, T>> for StatsOperand<'a, T> {
    fn from(value: ArrayView<'a, T>) -> Self {
        Self::View(value)
    }
}

impl<'a, T: Numeric> From<&'a ArrayView<'a, T>> for StatsOperand<'a, T> {
    fn from(value: &'a ArrayView<'a, T>) -> Self {
        Self::View(value.clone())
    }
}

impl<'a, T: Numeric> StatsOperand<'a, T> {
    pub(crate) fn shape(&self) -> &[usize] {
        match self {
            Self::Array(array) => array.shape(),
            Self::View(view) => view.shape(),
        }
    }

    pub(crate) fn ndim(&self) -> usize {
        self.shape().len()
    }

    pub(crate) fn len(&self) -> AtlasStatsResult<usize> {
        Ok(checked_element_count(self.shape())?)
    }

    pub(crate) fn dense_slice(&self) -> Option<&[T]> {
        match self {
            Self::Array(array) => Some(array.dense_slice()),
            Self::View(view) => view.dense_slice(),
        }
    }

    pub(crate) fn iter(&'a self) -> StatsOperandIter<'a, T> {
        match self {
            Self::Array(array) => StatsOperandIter::Array(array.iter()),
            Self::View(view) => StatsOperandIter::View(view.iter()),
        }
    }
}
