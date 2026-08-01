mod dot;
mod matmul;

use atlas_ndarray::Numeric;

use crate::core::LinalgOperand;

pub use dot::dot;
pub use matmul::matmul;

#[derive(Clone, Copy)]
pub(super) struct VectorRef<'a, T: Numeric> {
    pub(super) data: &'a [T],
    pub(super) offset: usize,
    pub(super) len: usize,
    pub(super) stride: usize,
}

impl<'a, T: Numeric> VectorRef<'a, T> {
    pub(super) fn is_contiguous(&self) -> bool {
        self.stride == 1
    }

    pub(super) fn value_at(&self, index: usize) -> T {
        self.data[self.offset + index * self.stride]
    }

    pub(super) fn contiguous_slice(&self) -> &'a [T] {
        debug_assert!(self.is_contiguous());
        &self.data[self.offset..self.offset + self.len]
    }
}

#[derive(Clone, Copy)]
pub(super) struct MatrixRef<'a, T: Numeric> {
    pub(super) data: &'a [T],
    pub(super) offset: usize,
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) row_stride: usize,
    pub(super) col_stride: usize,
}

impl<'a, T: Numeric> MatrixRef<'a, T> {
    pub(super) fn is_row_major_contiguous(&self) -> bool {
        self.col_stride == 1 && self.row_stride == self.cols
    }

    pub(super) fn is_col_major_contiguous(&self) -> bool {
        self.row_stride == 1 && self.col_stride == self.rows
    }

    pub(super) fn value_at(&self, row: usize, col: usize) -> T {
        self.data[self.offset + row * self.row_stride + col * self.col_stride]
    }

    pub(super) fn contiguous_row_slice(&self, row: usize) -> &'a [T] {
        debug_assert!(self.is_row_major_contiguous());
        let start = self.offset + row * self.row_stride;

        &self.data[start..start + self.cols]
    }

    pub(super) fn contiguous_col_slice(&self, col: usize) -> &'a [T] {
        debug_assert!(self.is_col_major_contiguous());
        let start = self.offset + col * self.col_stride;

        &self.data[start..start + self.rows]
    }
}

pub(super) fn vector_ref<'a, T: Numeric>(operand: &'a LinalgOperand<'a, T>) -> VectorRef<'a, T> {
    VectorRef {
        data: operand.data(),
        offset: operand.offset(),
        len: operand.shape()[0],
        stride: operand.strides()[0],
    }
}

pub(super) fn matrix_ref<'a, T: Numeric>(operand: &'a LinalgOperand<'a, T>) -> MatrixRef<'a, T> {
    MatrixRef {
        data: operand.data(),
        offset: operand.offset(),
        rows: operand.shape()[0],
        cols: operand.shape()[1],
        row_stride: operand.strides()[0],
        col_stride: operand.strides()[1],
    }
}
