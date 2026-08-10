use crate::{AsArray, NDArray, Numeric, internal::layout::dense_storage_slice, view::ArrayView};

pub trait OperandMetadata<T: Numeric>: sealed::Sealed {
    fn data(&self) -> &[T];

    fn offset(&self) -> usize;

    fn shape(&self) -> &[usize];

    fn strides(&self) -> &[usize];

    fn ndim(&self) -> usize {
        self.shape().len()
    }

    fn dense_slice(&self) -> Option<&[T]> {
        dense_storage_slice(self.data(), self.offset(), self.shape(), self.strides())
    }
}

impl<T: Numeric> OperandMetadata<T> for NDArray<T> {
    fn data(&self) -> &[T] {
        self.data()
    }

    fn offset(&self) -> usize {
        0
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn strides(&self) -> &[usize] {
        self.strides()
    }
}

impl<T: Numeric> OperandMetadata<T> for ArrayView<'_, T> {
    fn data(&self) -> &[T] {
        self.data()
    }

    fn offset(&self) -> usize {
        self.offset()
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn strides(&self) -> &[usize] {
        self.strides()
    }
}

impl<T: Numeric> OperandMetadata<T> for AsArray<'_, T> {
    fn data(&self) -> &[T] {
        match self {
            AsArray::Borrowed(view) => view.data(),
            AsArray::Owned(array) => array.data(),
        }
    }

    fn offset(&self) -> usize {
        match self {
            AsArray::Borrowed(view) => view.offset(),
            AsArray::Owned(_) => 0,
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            AsArray::Borrowed(view) => view.shape(),
            AsArray::Owned(array) => array.shape(),
        }
    }

    fn strides(&self) -> &[usize] {
        match self {
            AsArray::Borrowed(view) => view.strides(),
            AsArray::Owned(array) => array.strides(),
        }
    }
}

mod sealed {
    use crate::{AsArray, NDArray, Numeric, view::ArrayView};

    pub trait Sealed {}

    impl<T: Numeric> Sealed for NDArray<T> {}
    impl<T: Numeric> Sealed for ArrayView<'_, T> {}
    impl<T: Numeric> Sealed for AsArray<'_, T> {}
}

#[cfg(test)]
mod tests {
    use crate::{AsArray, NDArray, OperandMetadata};

    #[test]
    fn operand_metadata_reports_owned_and_view_metadata_consistently() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().transpose();
        let asarray = AsArray::Borrowed(view.clone());

        assert_eq!(OperandMetadata::shape(&array), &[2, 3]);
        assert_eq!(OperandMetadata::strides(&array), &[3, 1]);
        assert_eq!(OperandMetadata::offset(&array), 0);
        assert_eq!(OperandMetadata::ndim(&array), 2);
        assert_eq!(OperandMetadata::dense_slice(&array), Some(array.data()));

        assert_eq!(OperandMetadata::shape(&view), &[3, 2]);
        assert_eq!(OperandMetadata::strides(&view), &[1, 3]);
        assert_eq!(OperandMetadata::offset(&view), 0);
        assert_eq!(OperandMetadata::ndim(&view), 2);
        assert_eq!(OperandMetadata::dense_slice(&view), Some(array.data()));

        assert_eq!(OperandMetadata::shape(&asarray), &[3, 2]);
        assert_eq!(OperandMetadata::strides(&asarray), &[1, 3]);
        assert_eq!(OperandMetadata::offset(&asarray), 0);
        assert_eq!(OperandMetadata::ndim(&asarray), 2);
        assert_eq!(OperandMetadata::dense_slice(&asarray), Some(array.data()));
    }
}
