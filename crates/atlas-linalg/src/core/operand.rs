use atlas_ndarray::{ArrayView, NDArray, Numeric};

#[derive(Clone, Debug)]
pub enum LinalgOperand<'a, T: Numeric> {
    Array(&'a NDArray<T>),
    View(ArrayView<'a, T>),
}

enum OperandRef<'operand, 'data, T: Numeric> {
    Array(&'operand NDArray<T>),
    View(&'operand ArrayView<'data, T>),
}

impl<'a, T: Numeric> From<&'a NDArray<T>> for LinalgOperand<'a, T> {
    fn from(value: &'a NDArray<T>) -> Self {
        Self::Array(value)
    }
}

impl<'a, T: Numeric> From<ArrayView<'a, T>> for LinalgOperand<'a, T> {
    fn from(value: ArrayView<'a, T>) -> Self {
        Self::View(value)
    }
}

impl<'a, T: Numeric> From<&'a ArrayView<'a, T>> for LinalgOperand<'a, T> {
    fn from(value: &'a ArrayView<'a, T>) -> Self {
        Self::View(value.clone())
    }
}

impl<'a, T: Numeric> LinalgOperand<'a, T> {
    fn as_ref<'operand>(&'operand self) -> OperandRef<'operand, 'a, T> {
        match self {
            Self::Array(array) => OperandRef::Array(array),
            Self::View(view) => OperandRef::View(view),
        }
    }

    pub(crate) fn data(&self) -> &[T] {
        self.as_ref().data()
    }

    pub(crate) fn offset(&self) -> usize {
        self.as_ref().offset()
    }

    pub(crate) fn shape(&self) -> &[usize] {
        self.as_ref().shape()
    }

    pub(crate) fn strides(&self) -> &[usize] {
        self.as_ref().strides()
    }

    pub(crate) fn ndim(&self) -> usize {
        self.as_ref().ndim()
    }
}

impl<'operand, 'data, T: Numeric> OperandRef<'operand, 'data, T> {
    fn data(self) -> &'operand [T] {
        match self {
            Self::Array(array) => array.data(),
            Self::View(view) => view.data(),
        }
    }

    fn offset(self) -> usize {
        match self {
            Self::Array(_) => 0,
            Self::View(view) => view.offset(),
        }
    }

    fn shape(self) -> &'operand [usize] {
        match self {
            Self::Array(array) => array.shape(),
            Self::View(view) => view.shape(),
        }
    }

    fn strides(self) -> &'operand [usize] {
        match self {
            Self::Array(array) => array.strides(),
            Self::View(view) => view.strides(),
        }
    }

    fn ndim(self) -> usize {
        self.shape().len()
    }
}
