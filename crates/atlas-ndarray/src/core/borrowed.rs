use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, NDArray,
    internal::{
        layout::{dense_storage_slice, is_contiguous_layout, is_fortran_contiguous_layout},
        materialize_contiguous_array,
        shape::checked_element_count,
        validate_view_invariants,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct BorrowedArray<'a, T: ArrayElement> {
    data: &'a [T],
    shape: &'a [usize],
    strides: &'a [usize],
    offset: usize,
}

impl<'a, T: ArrayElement> BorrowedArray<'a, T> {
    pub fn from_parts(
        data: &'a [T],
        shape: &'a [usize],
        strides: &'a [usize],
        offset: usize,
    ) -> AtlasNdResult<Self> {
        validate_view_invariants(data.len(), offset, shape, strides)?;
        validate_positive_strides(shape, strides)?;

        Ok(Self { data, shape, strides, offset })
    }

    pub fn data(&self) -> &'a [T] {
        self.data
    }

    pub const fn offset(&self) -> usize {
        self.offset
    }

    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    pub fn strides(&self) -> &'a [usize] {
        self.strides
    }

    pub fn len(&self) -> usize {
        checked_element_count(self.shape)
            .expect("borrowed array construction validates the element count")
    }

    pub fn size(&self) -> usize {
        self.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub const fn is_owned(&self) -> bool {
        false
    }

    pub fn is_c_contiguous(&self) -> bool {
        is_contiguous_layout(self.shape, self.strides)
    }

    pub fn is_contiguous(&self) -> bool {
        self.is_c_contiguous()
    }

    pub fn is_fortran_contiguous(&self) -> bool {
        is_fortran_contiguous_layout(self.shape, self.strides)
    }

    pub fn dense_slice(&self) -> Option<&'a [T]> {
        dense_storage_slice(self.data, self.offset, self.shape, self.strides)
    }

    pub fn to_owned(&self) -> NDArray<T> {
        materialize_contiguous_array(self)
    }
}

fn validate_positive_strides(shape: &[usize], strides: &[usize]) -> AtlasNdResult<()> {
    let len = checked_element_count(shape)?;
    if len > 1
        && shape.iter().zip(strides).any(|(&dimension, &stride)| dimension > 1 && stride == 0)
    {
        return Err(AtlasNdError::InvalidShape);
    }

    Ok(())
}
