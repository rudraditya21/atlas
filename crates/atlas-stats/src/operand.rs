use atlas_ndarray::{ArrayView, NDArray, Numeric};

#[derive(Clone, Debug)]
pub enum StatsOperand<'a, T: Numeric> {
    Array(&'a NDArray<T>),
    View(ArrayView<'a, T>),
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
    pub(crate) fn data(&self) -> &[T] {
        match self {
            Self::Array(array) => array.data(),
            Self::View(view) => view.data(),
        }
    }

    pub(crate) fn offset(&self) -> usize {
        match self {
            Self::Array(_) => 0,
            Self::View(view) => view.offset(),
        }
    }

    pub(crate) fn shape(&self) -> &[usize] {
        match self {
            Self::Array(array) => array.shape(),
            Self::View(view) => view.shape(),
        }
    }

    pub(crate) fn strides(&self) -> &[usize] {
        match self {
            Self::Array(array) => array.strides(),
            Self::View(view) => view.strides(),
        }
    }

    pub(crate) fn ndim(&self) -> usize {
        self.shape().len()
    }

    pub(crate) fn len(&self) -> usize {
        if self.shape().is_empty() { 1 } else { self.shape().iter().product() }
    }
}
