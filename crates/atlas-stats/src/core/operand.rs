use std::slice::Iter;

use atlas_ndarray::{
    ArrayView, ArrayViewIter, NDArray, Numeric, OperandMetadata, checked_element_count,
    try_for_each_logical_span, try_for_each_logical_span_pair,
};

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

    pub(crate) fn iter<'operand>(&'operand self) -> StatsOperandIter<'operand, T> {
        self.as_ref().iter()
    }

    pub(crate) fn try_for_each_span<E, F>(&self, f: F) -> Result<(), E>
    where
        F: FnMut(&[T]) -> Result<(), E>,
    {
        match self {
            Self::Array(array) => try_for_each_logical_span(*array, f),
            Self::View(view) => try_for_each_logical_span(view, f),
        }
    }

    pub(crate) fn try_for_each_span_pair<E, F>(&self, other: &Self, f: F) -> Result<(), E>
    where
        F: FnMut(&[T], &[T]) -> Result<(), E>,
    {
        match (self, other) {
            (Self::Array(lhs), Self::Array(rhs)) => try_for_each_logical_span_pair(*lhs, *rhs, f),
            (Self::Array(lhs), Self::View(rhs)) => try_for_each_logical_span_pair(*lhs, rhs, f),
            (Self::View(lhs), Self::Array(rhs)) => try_for_each_logical_span_pair(lhs, *rhs, f),
            (Self::View(lhs), Self::View(rhs)) => try_for_each_logical_span_pair(lhs, rhs, f),
        }
    }
}

impl<'operand, 'data, T: Numeric> OperandRef<'operand, 'data, T> {
    fn metadata(self) -> &'operand dyn OperandMetadata<T> {
        match self {
            Self::Array(array) => array,
            Self::View(view) => view,
        }
    }

    fn shape(self) -> &'operand [usize] {
        self.metadata().shape()
    }

    fn ndim(self) -> usize {
        self.metadata().ndim()
    }

    fn iter(self) -> StatsOperandIter<'operand, T> {
        match self {
            Self::Array(array) => StatsOperandIter::Array(array.iter()),
            Self::View(view) => StatsOperandIter::View(view.iter()),
        }
    }
}
