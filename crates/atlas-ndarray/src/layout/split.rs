use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, AxisIndex, NDArray, core::axis::normalize_axis,
    view::ArrayView,
};

impl<T: ArrayElement> NDArray<T> {
    /// Splits an axis into equally sized metadata-only views.
    pub fn split<A: AxisIndex>(
        &self,
        sections: usize,
        axis: A,
    ) -> AtlasNdResult<Vec<ArrayView<'_, T>>> {
        self.view().split(sections, axis)
    }

    /// Splits an axis into near-even metadata-only views.
    pub fn array_split<A: AxisIndex>(
        &self,
        sections: usize,
        axis: A,
    ) -> AtlasNdResult<Vec<ArrayView<'_, T>>> {
        self.view().array_split(sections, axis)
    }

    /// Splits an axis at the provided indices into metadata-only views.
    pub fn split_at_indices<A: AxisIndex>(
        &self,
        indices: &[usize],
        axis: A,
    ) -> AtlasNdResult<Vec<ArrayView<'_, T>>> {
        self.view().split_at_indices(indices, axis)
    }
}

impl<'a, T: ArrayElement> ArrayView<'a, T> {
    /// Splits an axis into equally sized metadata-only views.
    ///
    /// `sections` must be non-zero and divide the selected axis length exactly.
    pub fn split<A: AxisIndex>(&self, sections: usize, axis: A) -> AtlasNdResult<Vec<Self>> {
        let axis = normalize_split_axis(axis, self.ndim())?;
        validate_sections(sections)?;
        let axis_len = self.shape[axis];
        if axis_len % sections != 0 {
            return Err(AtlasNdError::InvalidArgument {
                op: "split",
                reason: "axis length must be divisible by sections",
            });
        }

        self.split_lengths(axis, std::iter::repeat_n(axis_len / sections, sections))
    }

    /// Splits an axis into near-even metadata-only views.
    ///
    /// The first views receive one extra element when the axis length is not evenly divisible.
    pub fn array_split<A: AxisIndex>(&self, sections: usize, axis: A) -> AtlasNdResult<Vec<Self>> {
        let axis = normalize_split_axis(axis, self.ndim())?;
        validate_sections(sections)?;
        let axis_len = self.shape[axis];
        let base = axis_len / sections;
        let remainder = axis_len % sections;
        let lengths = (0..sections).map(|index| base + usize::from(index < remainder));

        self.split_lengths(axis, lengths)
    }

    /// Splits an axis at the provided indices into metadata-only views.
    pub fn split_at_indices<A: AxisIndex>(
        &self,
        indices: &[usize],
        axis: A,
    ) -> AtlasNdResult<Vec<Self>> {
        let axis = normalize_split_axis(axis, self.ndim())?;
        let axis_len = self.shape[axis];
        let mut start = 0;
        let mut views = Vec::with_capacity(indices.len() + 1);

        for &end in indices.iter().chain(std::iter::once(&axis_len)) {
            if end < start || end > axis_len {
                return Err(AtlasNdError::InvalidArgument {
                    op: "split",
                    reason: "split indices must be sorted and within the selected axis",
                });
            }

            let mut starts = vec![0; self.ndim()];
            let mut shape = self.shape.clone();
            starts[axis] = start;
            shape[axis] = end - start;
            views.push(self.slice(starts, shape)?);
            start = end;
        }

        Ok(views)
    }

    fn split_lengths(
        &self,
        axis: usize,
        lengths: impl IntoIterator<Item = usize>,
    ) -> AtlasNdResult<Vec<Self>> {
        let mut offset = 0;
        let mut views = Vec::new();

        for length in lengths {
            let mut starts = vec![0; self.ndim()];
            let mut shape = self.shape.clone();
            starts[axis] = offset;
            shape[axis] = length;
            views.push(self.slice(starts, shape)?);
            offset += length;
        }

        Ok(views)
    }
}

fn normalize_split_axis<A: AxisIndex>(axis: A, ndim: usize) -> AtlasNdResult<usize> {
    normalize_axis(axis, ndim)
}

fn validate_sections(sections: usize) -> AtlasNdResult<()> {
    if sections == 0 {
        return Err(AtlasNdError::InvalidArgument {
            op: "split",
            reason: "sections must be greater than zero",
        });
    }

    Ok(())
}
