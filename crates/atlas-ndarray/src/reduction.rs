use num_traits::ToPrimitive;

use super::{
    array::NDArray,
    error::{AtlasNdError, AtlasNdResult},
    stride::element_count,
    traits::Numeric,
    traversal::{for_each_value, try_for_each_value},
    view::ArrayView,
};

impl<T: Numeric> NDArray<T> {
    pub fn sum(&self) -> T {
        if self.is_contiguous() {
            return sum_contiguous(&self.data);
        }

        sum_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn prod(&self) -> T {
        if self.is_contiguous() {
            return prod_contiguous(&self.data);
        }

        prod_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn min(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        if self.is_contiguous() {
            return min_contiguous(&self.data, "min");
        }

        min_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn max(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        if self.is_contiguous() {
            return max_contiguous(&self.data, "max");
        }

        max_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn mean(&self) -> AtlasNdResult<f64>
    where
        T: ToPrimitive,
    {
        if self.is_contiguous() {
            return mean_contiguous(&self.data, "mean");
        }

        mean_all(&self.data, 0, &self.shape, &self.strides)
    }
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn sum(&self) -> T {
        if self.is_contiguous() {
            return sum_contiguous(self.contiguous_slice());
        }

        sum_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn prod(&self) -> T {
        if self.is_contiguous() {
            return prod_contiguous(self.contiguous_slice());
        }

        prod_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn min(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        if self.is_contiguous() {
            return min_contiguous(self.contiguous_slice(), "min");
        }

        min_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn max(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        if self.is_contiguous() {
            return max_contiguous(self.contiguous_slice(), "max");
        }

        max_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn mean(&self) -> AtlasNdResult<f64>
    where
        T: ToPrimitive,
    {
        if self.is_contiguous() {
            return mean_contiguous(self.contiguous_slice(), "mean");
        }

        mean_all(self.data, self.offset, &self.shape, &self.strides)
    }
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    fn contiguous_slice(&self) -> &[T] {
        debug_assert!(self.is_contiguous());
        let len = element_count(&self.shape);
        &self.data[self.offset..self.offset + len]
    }
}

fn sum_contiguous<T: Numeric>(values: &[T]) -> T {
    let mut total = T::zero();

    for &value in values {
        total += value;
    }

    total
}

fn prod_contiguous<T: Numeric>(values: &[T]) -> T {
    let mut total = T::one();

    for &value in values {
        total *= value;
    }

    total
}

fn min_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut iter = values.iter().copied();
    let mut minimum = iter.next().ok_or(AtlasNdError::EmptyReduction { op })?;

    for value in iter {
        if value < minimum {
            minimum = value;
        }
    }

    Ok(minimum)
}

fn max_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut iter = values.iter().copied();
    let mut maximum = iter.next().ok_or(AtlasNdError::EmptyReduction { op })?;

    for value in iter {
        if value > maximum {
            maximum = value;
        }
    }

    Ok(maximum)
}

fn mean_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
{
    if values.is_empty() {
        return Err(AtlasNdError::EmptyReduction { op });
    }

    let mut total = 0.0_f64;

    for &value in values {
        total += value
            .to_f64()
            .ok_or(AtlasNdError::NumericConversionFailed { op })?;
    }

    Ok(total / values.len() as f64)
}

fn sum_all<T: Numeric>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> T {
    let mut total = T::zero();
    for_each_value(data, offset, shape, strides, |value| {
        total += *value;
    });
    total
}

fn prod_all<T: Numeric>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> T {
    let mut total = T::one();
    for_each_value(data, offset, shape, strides, |value| {
        total *= *value;
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
            Some(current) if current < *value => current,
            Some(_) | None => *value,
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
            Some(current) if current > *value => current,
            Some(_) | None => *value,
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
