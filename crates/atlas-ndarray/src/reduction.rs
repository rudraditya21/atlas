use num_traits::ToPrimitive;

use super::{
    array::NDArray,
    error::{AtlasNdError, AtlasNdResult},
    stride::{compute_strides, element_count},
    traits::Numeric,
    view::ArrayView,
};

impl<T: Numeric> NDArray<T> {
    pub fn sum(&self) -> T {
        sum_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn prod(&self) -> T {
        prod_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn min(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        min_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn max(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        max_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn mean(&self) -> AtlasNdResult<f64>
    where
        T: ToPrimitive,
    {
        mean_all(&self.data, 0, &self.shape, &self.strides)
    }
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn sum(&self) -> T {
        sum_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn prod(&self) -> T {
        prod_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn min(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        min_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn max(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        max_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn mean(&self) -> AtlasNdResult<f64>
    where
        T: ToPrimitive,
    {
        mean_all(self.data, self.offset, &self.shape, &self.strides)
    }
}

fn sum_all<T: Numeric>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> T {
    let mut total = T::zero();
    for_each_value(data, offset, shape, strides, |value| {
        total += value;
    });
    total
}

fn prod_all<T: Numeric>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> T {
    let mut total = T::one();
    for_each_value(data, offset, shape, strides, |value| {
        total *= value;
    });
    total
}

fn min_all<T>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut minimum = None;

    for_each_value(data, offset, shape, strides, |value| {
        minimum = Some(match minimum {
            Some(current) if current < value => current,
            Some(_) | None => value,
        });
    });

    minimum.ok_or(AtlasNdError::EmptyReduction { op: "min" })
}

fn max_all<T>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut maximum = None;

    for_each_value(data, offset, shape, strides, |value| {
        maximum = Some(match maximum {
            Some(current) if current > value => current,
            Some(_) | None => value,
        });
    });

    maximum.ok_or(AtlasNdError::EmptyReduction { op: "max" })
}

fn mean_all<T>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
{
    let len = element_count(shape);
    if len == 0 {
        return Err(AtlasNdError::EmptyReduction { op: "mean" });
    }

    let mut total = 0.0_f64;
    try_for_each_value(data, offset, shape, strides, |value| {
        total += value
            .to_f64()
            .ok_or(AtlasNdError::NumericConversionFailed { op: "mean" })?;
        Ok(())
    })?;

    Ok(total / len as f64)
}

fn for_each_value<T, F>(data: &[T], offset: usize, shape: &[usize], strides: &[usize], mut f: F)
where
    T: Numeric,
    F: FnMut(T),
{
    let len = element_count(shape);
    if len == 0 {
        return;
    }

    if is_contiguous_layout(shape, strides) {
        let end = offset + len;
        for &value in &data[offset..end] {
            f(value);
        }
        return;
    }

    for linear_index in 0..len {
        let physical_offset = offset_from_linear_index(linear_index, offset, shape, strides);
        f(data[physical_offset]);
    }
}

fn try_for_each_value<T, F>(
    data: &[T],
    offset: usize,
    shape: &[usize],
    strides: &[usize],
    mut f: F,
) -> AtlasNdResult<()>
where
    T: Numeric,
    F: FnMut(T) -> AtlasNdResult<()>,
{
    let len = element_count(shape);
    if len == 0 {
        return Ok(());
    }

    if is_contiguous_layout(shape, strides) {
        let end = offset + len;
        for &value in &data[offset..end] {
            f(value)?;
        }
        return Ok(());
    }

    for linear_index in 0..len {
        let physical_offset = offset_from_linear_index(linear_index, offset, shape, strides);
        f(data[physical_offset])?;
    }

    Ok(())
}

fn is_contiguous_layout(shape: &[usize], strides: &[usize]) -> bool {
    debug_assert_eq!(shape.len(), strides.len());
    strides == compute_strides(shape)
}

fn offset_from_linear_index(
    mut linear_index: usize,
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> usize {
    debug_assert_eq!(shape.len(), strides.len());

    let mut offset = base_offset;

    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        let coordinate = if dim == 0 { 0 } else { linear_index % dim };
        linear_index = if dim == 0 { 0 } else { linear_index / dim };
        offset += coordinate * strides[axis];
    }

    offset
}

#[cfg(test)]
mod tests {
    use crate::{array::NDArray, error::AtlasNdError};

    #[test]
    fn whole_array_reductions_work_for_contiguous_arrays() {
        let array = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();

        assert_eq!(array.sum(), 10);
        assert_eq!(array.prod(), 24);
        assert_eq!(array.min().unwrap(), 1);
        assert_eq!(array.max().unwrap(), 4);
        assert_eq!(array.mean().unwrap(), 2.5);
    }

    #[test]
    fn whole_array_reductions_work_for_non_contiguous_views() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().slice(&[0, 1], vec![2, 2]).unwrap();

        assert_eq!(view.sum(), 12);
        assert_eq!(view.prod(), 40);
        assert_eq!(view.min().unwrap(), 1);
        assert_eq!(view.max().unwrap(), 5);
        assert_eq!(view.mean().unwrap(), 3.0);
    }

    #[test]
    fn sum_and_prod_use_identity_for_empty_arrays() {
        let array = NDArray::<i32>::new(vec![0, 3], 7);

        assert_eq!(array.sum(), 0);
        assert_eq!(array.prod(), 1);
    }

    #[test]
    fn min_max_and_mean_reject_empty_arrays() {
        let array = NDArray::<i32>::new(vec![0, 3], 7);

        assert_eq!(array.min().unwrap_err(), AtlasNdError::EmptyReduction { op: "min" });
        assert_eq!(array.max().unwrap_err(), AtlasNdError::EmptyReduction { op: "max" });
        assert_eq!(
            array.mean().unwrap_err(),
            AtlasNdError::EmptyReduction { op: "mean" }
        );
    }
}
