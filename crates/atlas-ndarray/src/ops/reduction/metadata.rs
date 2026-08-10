use crate::{
    AtlasNdResult, AxisIndex,
    core::axis::normalize_axis,
    internal::layout::{LayoutKind, is_contiguous_layout},
    layout::element_count,
};

pub(super) struct WholeReductionMetadata {
    pub(super) len: usize,
}

impl WholeReductionMetadata {
    pub(super) fn from_shape(shape: &[usize]) -> Self {
        Self { len: element_count(shape) }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub(super) struct ReductionOutputMetadata {
    pub(super) shape: Vec<usize>,
    pub(super) outer_strides: Vec<usize>,
    pub(super) len: usize,
}

impl ReductionOutputMetadata {
    fn new(shape: Vec<usize>, outer_strides: Vec<usize>) -> Self {
        let len = element_count(&shape);
        Self { shape, outer_strides, len }
    }
}

pub(super) struct AxisReductionMetadata {
    pub(super) output: ReductionOutputMetadata,
    pub(super) axis_stride: usize,
    pub(super) axis_len: usize,
    pub(super) contiguous_outer_len: usize,
    pub(super) contiguous_inner_len: usize,
    pub(super) source_layout: LayoutKind,
    pub(super) axis_layout: LayoutKind,
}

pub(super) fn axis_reduction_metadata(
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<AxisReductionMetadata> {
    axis_reduction_metadata_with(shape, strides, axis, false)
}

pub(super) fn axis_reduction_metadata_keepdims(
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<AxisReductionMetadata> {
    axis_reduction_metadata_with(shape, strides, axis, true)
}

fn axis_reduction_metadata_with(
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
    keepdims: bool,
) -> AtlasNdResult<AxisReductionMetadata> {
    let axis = normalize_axis(axis, shape.len())?;

    let mut output_shape =
        Vec::with_capacity(if keepdims { shape.len() } else { shape.len().saturating_sub(1) });
    let mut outer_strides =
        Vec::with_capacity(if keepdims { strides.len() } else { strides.len().saturating_sub(1) });

    for (current_axis, (&dim, &stride)) in shape.iter().zip(strides.iter()).enumerate() {
        if current_axis == axis {
            if keepdims {
                output_shape.push(1);
                outer_strides.push(stride);
            }
            continue;
        }

        output_shape.push(dim);
        outer_strides.push(stride);
    }

    let output = ReductionOutputMetadata::new(output_shape, outer_strides);

    Ok(AxisReductionMetadata {
        output,
        axis_stride: strides[axis],
        axis_len: shape[axis],
        contiguous_outer_len: element_count(&shape[..axis]),
        contiguous_inner_len: element_count(&shape[axis + 1..]),
        source_layout: if is_contiguous_layout(shape, strides) {
            LayoutKind::Contiguous
        } else {
            LayoutKind::Strided
        },
        axis_layout: if strides[axis] == 1 { LayoutKind::Contiguous } else { LayoutKind::Strided },
    })
}
