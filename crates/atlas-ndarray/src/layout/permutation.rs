use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, AxisIndex, NDArray, core::axis::normalize_axis,
    view::ArrayView,
};

impl<T: ArrayElement> NDArray<T> {
    /// Returns a metadata-only view whose axes follow `axes`.
    pub fn permute_axes<A, P>(&self, axes: P) -> AtlasNdResult<ArrayView<'_, T>>
    where
        A: AxisIndex,
        P: AsRef<[A]>,
    {
        self.view().permute_axes(axes)
    }

    /// Returns a metadata-only view with two axes exchanged.
    pub fn swap_axes<A: AxisIndex>(&self, left: A, right: A) -> AtlasNdResult<ArrayView<'_, T>> {
        self.view().swap_axes(left, right)
    }
}

impl<'a, T: ArrayElement> ArrayView<'a, T> {
    /// Returns a metadata-only view whose axes follow `axes`.
    ///
    /// Axes must be a complete permutation of the current axes. Negative axes are normalized
    /// relative to the view rank.
    pub fn permute_axes<A, P>(self, axes: P) -> AtlasNdResult<Self>
    where
        A: AxisIndex,
        P: AsRef<[A]>,
    {
        let axes = normalize_permutation(axes.as_ref(), self.ndim())?;
        let shape = axes.iter().map(|&axis| self.shape[axis]).collect();
        let strides = axes.iter().map(|&axis| self.strides[axis]).collect();

        ArrayView::from_parts(self.data, self.offset, shape, strides)
    }

    /// Returns a metadata-only view with two axes exchanged.
    pub fn swap_axes<A: AxisIndex>(self, left: A, right: A) -> AtlasNdResult<Self> {
        let ndim = self.ndim();
        let left = normalize_axis(left, ndim)?;
        let right = normalize_axis(right, ndim)?;
        let mut axes: Vec<_> = (0..ndim).collect();
        axes.swap(left, right);

        self.permute_axes(axes)
    }
}

fn normalize_permutation<A: AxisIndex>(axes: &[A], ndim: usize) -> AtlasNdResult<Vec<usize>> {
    if axes.len() != ndim {
        return Err(AtlasNdError::DimensionMismatch { expected: ndim, actual: axes.len() });
    }

    let mut normalized = Vec::with_capacity(ndim);
    let mut seen = vec![false; ndim];

    for &axis in axes {
        let axis = normalize_axis(axis, ndim)?;
        if std::mem::replace(&mut seen[axis], true) {
            return Err(AtlasNdError::InvalidArgument {
                op: "permute_axes",
                reason: "axes must be a permutation",
            });
        }
        normalized.push(axis);
    }

    Ok(normalized)
}
