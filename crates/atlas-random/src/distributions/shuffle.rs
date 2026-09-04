use atlas_ndarray::{ArrayElement, AtlasNdError, AxisIndex, NDArray, OperandMetadata};

use crate::{core::AtlasRandomResult, rng::random_source::RandomSource};

/// Returns `input` with one seeded permutation applied along `axis`.
///
/// The output preserves the logical shape and contains every input element exactly once.
pub fn shuffle_axis<T, O, A, R>(input: &O, axis: A, rng: &mut R) -> AtlasRandomResult<NDArray<T>>
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
    A: AxisIndex,
    R: RandomSource,
{
    let axis = normalize_axis(axis, input.ndim())?;
    let mut permutation = (0..input.shape()[axis]).collect::<Vec<_>>();
    rng.shuffle(&mut permutation);

    let mut data = Vec::with_capacity(input.shape().iter().product());
    for linear_index in 0..input.shape().iter().product() {
        data.push(input.data()[source_offset(input, axis, &permutation, linear_index)]);
    }

    NDArray::from_shape_vec(input.shape().to_vec(), data).map_err(Into::into)
}

fn normalize_axis<A: AxisIndex>(axis: A, ndim: usize) -> Result<usize, AtlasNdError> {
    let axis = axis.try_into_i64().ok_or(AtlasNdError::InvalidAxis { axis: i64::MAX, ndim })?;
    let normalized = if axis < 0 { ndim as i64 + axis } else { axis };
    if normalized < 0 || normalized >= ndim as i64 {
        Err(AtlasNdError::InvalidAxis { axis, ndim })
    } else {
        Ok(normalized as usize)
    }
}

fn source_offset<T: ArrayElement, O: OperandMetadata<T> + ?Sized>(
    input: &O,
    shuffled_axis: usize,
    permutation: &[usize],
    mut linear_index: usize,
) -> usize {
    let mut offset = input.offset();
    for axis in (0..input.ndim()).rev() {
        let coordinate = linear_index % input.shape()[axis];
        linear_index /= input.shape()[axis];
        let coordinate = if axis == shuffled_axis { permutation[coordinate] } else { coordinate };
        offset += coordinate * input.strides()[axis];
    }
    offset
}
