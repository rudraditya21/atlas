use std::slice::{Iter, IterMut};
use std::vec::IntoIter;

use super::{array::NDArray, traits::Numeric, view::ArrayView};

pub struct ArrayViewIter<'a, T> {
    inner: super::traversal::ValueIter<'a, T>,
}

impl<'a, T> Iterator for ArrayViewIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<T: Numeric> NDArray<T> {
    pub fn iter(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.data.iter_mut()
    }
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn iter(&'a self) -> ArrayViewIter<'a, T> {
        ArrayViewIter {
            inner: super::traversal::value_iter(self.data, self.offset, &self.shape, &self.strides),
        }
    }
}

impl<'a, T: Numeric> IntoIterator for &'a NDArray<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T: Numeric> IntoIterator for &'a mut NDArray<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

impl<T: Numeric> IntoIterator for NDArray<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T: Numeric> IntoIterator for &'a ArrayView<'a, T> {
    type Item = &'a T;
    type IntoIter = ArrayViewIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use crate::array::NDArray;

    #[test]
    fn array_view_iterates_contiguous_views_in_logical_order() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let collected: Vec<_> = array.view().iter().copied().collect();

        assert_eq!(collected, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn array_view_iterates_strided_views_in_logical_order() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let collected: Vec<_> = array.view().transpose().iter().copied().collect();

        assert_eq!(collected, vec![0, 3, 1, 4, 2, 5]);
    }
}
