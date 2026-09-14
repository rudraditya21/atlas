use atlas_ndarray::{ArrayView, NDArray, Numeric, OperandMetadata};

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

impl<'a, T: Numeric> From<&LinalgOperand<'a, T>> for LinalgOperand<'a, T> {
    fn from(value: &LinalgOperand<'a, T>) -> Self {
        value.clone()
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

    pub(crate) fn into_view(self) -> ArrayView<'a, T> {
        match self {
            Self::Array(array) => array.view(),
            Self::View(view) => view,
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

    fn data(self) -> &'operand [T] {
        self.metadata().data()
    }

    fn offset(self) -> usize {
        self.metadata().offset()
    }

    fn shape(self) -> &'operand [usize] {
        self.metadata().shape()
    }

    fn strides(self) -> &'operand [usize] {
        self.metadata().strides()
    }

    fn ndim(self) -> usize {
        self.metadata().ndim()
    }
}
