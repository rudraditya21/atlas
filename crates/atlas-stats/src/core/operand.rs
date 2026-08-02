use std::slice::Iter;

use atlas_ndarray::{ArrayView, ArrayViewIter, NDArray, Numeric, checked_element_count};

use crate::core::error::AtlasStatsResult;

#[derive(Clone, Debug)]
pub enum StatsOperand<'a, T: Numeric> {
    Array(&'a NDArray<T>),
    View(ArrayView<'a, T>),
}

enum OperandRef<'operand, 'data, T: Numeric> {
    Array(&'operand NDArray<T>),
    View(&'operand ArrayView<'data, T>),
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
    fn as_ref<'operand>(&'operand self) -> OperandRef<'operand, 'a, T> {
        match self {
            Self::Array(array) => OperandRef::Array(array),
            Self::View(view) => OperandRef::View(view),
        }
    }

    pub(crate) fn shape(&self) -> &[usize] {
        self.as_ref().shape()
    }

    pub(crate) fn ndim(&self) -> usize {
        self.as_ref().ndim()
    }

    pub(crate) fn len(&self) -> AtlasStatsResult<usize> {
        Ok(checked_element_count(self.shape())?)
    }

    pub(crate) fn dense_slice(&self) -> Option<&[T]> {
        self.as_ref().dense_slice()
    }

    pub(crate) fn iter<'operand>(&'operand self) -> StatsOperandIter<'operand, T> {
        self.as_ref().iter()
    }
}

impl<'operand, 'data, T: Numeric> OperandRef<'operand, 'data, T> {
    fn shape(self) -> &'operand [usize] {
        match self {
            Self::Array(array) => array.shape(),
            Self::View(view) => view.shape(),
        }
    }

    fn ndim(self) -> usize {
        self.shape().len()
    }

    fn dense_slice(self) -> Option<&'operand [T]> {
        match self {
            Self::Array(array) => Some(array.dense_slice()),
            Self::View(view) => view.dense_slice(),
        }
    }

    fn iter(self) -> StatsOperandIter<'operand, T> {
        match self {
            Self::Array(array) => StatsOperandIter::Array(array.iter()),
            Self::View(view) => StatsOperandIter::View(view.iter()),
        }
    }
}
