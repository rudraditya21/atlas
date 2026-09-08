use atlas_ndarray::Numeric;

use crate::{core::LinalgOperand, internal::simd};

#[derive(Clone, Copy)]
pub(crate) struct VectorRef<'a, T: Numeric> {
    pub(crate) data: &'a [T],
    pub(crate) offset: usize,
    pub(crate) len: usize,
    pub(crate) stride: usize,
}

impl<'a, T: Numeric> VectorRef<'a, T> {
    pub(crate) fn is_contiguous(&self) -> bool {
        self.stride == 1
    }

    pub(crate) fn value_at(&self, index: usize) -> T {
        self.data[self.offset + index * self.stride]
    }

    pub(crate) fn contiguous_slice(&self) -> &'a [T] {
        debug_assert!(self.is_contiguous());
        &self.data[self.offset..self.offset + self.len]
    }
}

#[derive(Clone, Copy)]
pub(crate) struct MatrixRef<'a, T: Numeric> {
    pub(crate) data: &'a [T],
    pub(crate) offset: usize,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) row_stride: usize,
    pub(crate) col_stride: usize,
}

impl<'a, T: Numeric> MatrixRef<'a, T> {
    pub(crate) fn is_row_major_contiguous(&self) -> bool {
        self.col_stride == 1 && self.row_stride == self.cols
    }

    pub(crate) fn is_col_major_contiguous(&self) -> bool {
        self.row_stride == 1 && self.col_stride == self.rows
    }

    pub(crate) fn value_at(&self, row: usize, col: usize) -> T {
        self.data[self.offset + row * self.row_stride + col * self.col_stride]
    }

    pub(crate) fn contiguous_row_slice(&self, row: usize) -> &'a [T] {
        debug_assert!(self.is_row_major_contiguous());
        let start = self.offset + row * self.row_stride;

        &self.data[start..start + self.cols]
    }

    pub(crate) fn row_major_region(&self) -> &'a [T] {
        debug_assert!(self.is_row_major_contiguous());
        let len = self.rows * self.cols;

        &self.data[self.offset..self.offset + len]
    }

    pub(crate) fn contiguous_col_slice(&self, col: usize) -> &'a [T] {
        debug_assert!(self.is_col_major_contiguous());
        let start = self.offset + col * self.col_stride;

        &self.data[start..start + self.rows]
    }
}

pub(crate) fn vector_ref<'a, T: Numeric>(operand: &'a LinalgOperand<'a, T>) -> VectorRef<'a, T> {
    VectorRef {
        data: operand.data(),
        offset: operand.offset(),
        len: operand.shape()[0],
        stride: operand.strides()[0],
    }
}

pub(crate) fn matrix_ref<'a, T: Numeric>(operand: &'a LinalgOperand<'a, T>) -> MatrixRef<'a, T> {
    MatrixRef {
        data: operand.data(),
        offset: operand.offset(),
        rows: operand.shape()[0],
        cols: operand.shape()[1],
        row_stride: operand.strides()[0],
        col_stride: operand.strides()[1],
    }
}

pub(crate) fn dot_kernel<T: Numeric>(lhs: VectorRef<'_, T>, rhs: VectorRef<'_, T>) -> T {
    if lhs.is_contiguous() && rhs.is_contiguous() {
        dot_contiguous(lhs.contiguous_slice(), rhs.contiguous_slice())
    } else {
        dot_strided(lhs, rhs)
    }
}

pub(crate) fn dot_contiguous<T: Numeric>(lhs: &[T], rhs: &[T]) -> T {
    simd::dot_contiguous(lhs, rhs)
}

pub(crate) fn dot_strided<T: Numeric>(lhs: VectorRef<'_, T>, rhs: VectorRef<'_, T>) -> T {
    let mut total = T::zero();
    let mut lhs_offset = lhs.offset;
    let mut rhs_offset = rhs.offset;

    for _ in 0..lhs.len {
        total += lhs.data[lhs_offset] * rhs.data[rhs_offset];
        lhs_offset += lhs.stride;
        rhs_offset += rhs.stride;
    }

    total
}
