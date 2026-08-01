use std::ops::{Add, Div, Mul, Sub};

use super::{
    array::NDArray,
    broadcast::broadcast_pair,
    error::AtlasNdResult,
    stride::element_count,
    traits::Numeric,
};

pub trait AddOperand<T: Numeric> {
    type Output;

    fn add_to(self, lhs: &NDArray<T>) -> Self::Output;
}

pub trait SubOperand<T: Numeric> {
    type Output;

    fn sub_from(self, lhs: &NDArray<T>) -> Self::Output;
}

pub trait MulOperand<T: Numeric> {
    type Output;

    fn mul_with(self, lhs: &NDArray<T>) -> Self::Output;
}

pub trait DivOperand<T: Numeric> {
    type Output;

    fn div_into(self, lhs: &NDArray<T>) -> Self::Output;
}

impl<T: Numeric> NDArray<T> {
    pub fn add<Rhs>(&self, rhs: Rhs) -> Rhs::Output
    where
        Rhs: AddOperand<T>,
    {
        rhs.add_to(self)
    }

    pub fn sub<Rhs>(&self, rhs: Rhs) -> Rhs::Output
    where
        Rhs: SubOperand<T>,
    {
        rhs.sub_from(self)
    }

    pub fn mul<Rhs>(&self, rhs: Rhs) -> Rhs::Output
    where
        Rhs: MulOperand<T>,
    {
        rhs.mul_with(self)
    }

    pub fn div<Rhs>(&self, rhs: Rhs) -> Rhs::Output
    where
        Rhs: DivOperand<T>,
    {
        rhs.div_into(self)
    }

    pub fn add_scalar(&self, scalar: T) -> Self {
        self.elementwise_scalar(scalar, |value, scalar| value + scalar)
    }

    pub fn sub_scalar(&self, scalar: T) -> Self {
        self.elementwise_scalar(scalar, |value, scalar| value - scalar)
    }

    pub fn mul_scalar(&self, scalar: T) -> Self {
        self.elementwise_scalar(scalar, |value, scalar| value * scalar)
    }

    pub fn div_scalar(&self, scalar: T) -> Self {
        self.elementwise_scalar(scalar, |value, scalar| value / scalar)
    }

    fn elementwise_binary<F>(&self, rhs: &Self, op: F) -> AtlasNdResult<Self>
    where
        F: Fn(T, T) -> T + Copy,
    {
        if self.is_contiguous() && rhs.is_contiguous() && self.shape == rhs.shape {
            return Ok(self.elementwise_binary_contiguous(rhs, op));
        }

        self.elementwise_binary_broadcast(rhs, op)
    }

    fn elementwise_binary_contiguous<F>(&self, rhs: &Self, op: F) -> Self
    where
        F: Fn(T, T) -> T + Copy,
    {
        let data = self
            .data
            .iter()
            .copied()
            .zip(rhs.data.iter().copied())
            .map(|(lhs, rhs)| op(lhs, rhs))
            .collect();

        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    fn elementwise_binary_broadcast<F>(&self, rhs: &Self, op: F) -> AtlasNdResult<Self>
    where
        F: Fn(T, T) -> T + Copy,
    {
        let metadata = broadcast_pair(&self.shape, &self.strides, &rhs.shape, &rhs.strides)?;
        let output_len = element_count(&metadata.shape);
        let mut data = Vec::with_capacity(output_len);

        for linear_index in 0..output_len {
            let (lhs_offset, rhs_offset) = broadcast_offsets(
                linear_index,
                &metadata.shape,
                &metadata.lhs_strides,
                &metadata.rhs_strides,
            );

            data.push(op(self.data[lhs_offset], rhs.data[rhs_offset]));
        }

        Ok(Self {
            data,
            shape: metadata.shape.clone(),
            strides: super::stride::compute_strides(&metadata.shape),
        })
    }

    fn elementwise_scalar<F>(&self, scalar: T, op: F) -> Self
    where
        F: Fn(T, T) -> T + Copy,
    {
        let data = self
            .data
            .iter()
            .copied()
            .map(|value| op(value, scalar))
            .collect();

        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<'a, T: Numeric> AddOperand<T> for &'a NDArray<T> {
    type Output = AtlasNdResult<NDArray<T>>;

    fn add_to(self, lhs: &NDArray<T>) -> Self::Output {
        lhs.elementwise_binary(self, |lhs, rhs| lhs + rhs)
    }
}

impl<T: Numeric> AddOperand<T> for T {
    type Output = NDArray<T>;

    fn add_to(self, lhs: &NDArray<T>) -> Self::Output {
        lhs.add_scalar(self)
    }
}

impl<'a, T: Numeric> SubOperand<T> for &'a NDArray<T> {
    type Output = AtlasNdResult<NDArray<T>>;

    fn sub_from(self, lhs: &NDArray<T>) -> Self::Output {
        lhs.elementwise_binary(self, |lhs, rhs| lhs - rhs)
    }
}

impl<T: Numeric> SubOperand<T> for T {
    type Output = NDArray<T>;

    fn sub_from(self, lhs: &NDArray<T>) -> Self::Output {
        lhs.sub_scalar(self)
    }
}

impl<'a, T: Numeric> MulOperand<T> for &'a NDArray<T> {
    type Output = AtlasNdResult<NDArray<T>>;

    fn mul_with(self, lhs: &NDArray<T>) -> Self::Output {
        lhs.elementwise_binary(self, |lhs, rhs| lhs * rhs)
    }
}

impl<T: Numeric> MulOperand<T> for T {
    type Output = NDArray<T>;

    fn mul_with(self, lhs: &NDArray<T>) -> Self::Output {
        lhs.mul_scalar(self)
    }
}

impl<'a, T: Numeric> DivOperand<T> for &'a NDArray<T> {
    type Output = AtlasNdResult<NDArray<T>>;

    fn div_into(self, lhs: &NDArray<T>) -> Self::Output {
        lhs.elementwise_binary(self, |lhs, rhs| lhs / rhs)
    }
}

impl<T: Numeric> DivOperand<T> for T {
    type Output = NDArray<T>;

    fn div_into(self, lhs: &NDArray<T>) -> Self::Output {
        lhs.div_scalar(self)
    }
}

fn broadcast_offsets(
    mut linear_index: usize,
    shape: &[usize],
    lhs_strides: &[usize],
    rhs_strides: &[usize],
) -> (usize, usize) {
    debug_assert_eq!(shape.len(), lhs_strides.len());
    debug_assert_eq!(shape.len(), rhs_strides.len());

    let mut lhs_offset = 0;
    let mut rhs_offset = 0;

    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        let coordinate = if dim == 0 { 0 } else { linear_index % dim };
        linear_index = if dim == 0 { 0 } else { linear_index / dim };

        lhs_offset += coordinate * lhs_strides[axis];
        rhs_offset += coordinate * rhs_strides[axis];
    }

    (lhs_offset, rhs_offset)
}

impl<T: Numeric> Add for &NDArray<T> {
    type Output = AtlasNdResult<NDArray<T>>;

    fn add(self, rhs: Self) -> Self::Output {
        NDArray::add(self, rhs)
    }
}

impl<T: Numeric> Sub for &NDArray<T> {
    type Output = AtlasNdResult<NDArray<T>>;

    fn sub(self, rhs: Self) -> Self::Output {
        NDArray::sub(self, rhs)
    }
}

impl<T: Numeric> Mul for &NDArray<T> {
    type Output = AtlasNdResult<NDArray<T>>;

    fn mul(self, rhs: Self) -> Self::Output {
        NDArray::mul(self, rhs)
    }
}

impl<T: Numeric> Div for &NDArray<T> {
    type Output = AtlasNdResult<NDArray<T>>;

    fn div(self, rhs: Self) -> Self::Output {
        NDArray::div(self, rhs)
    }
}

impl<T: Numeric> Add<T> for &NDArray<T> {
    type Output = NDArray<T>;

    fn add(self, rhs: T) -> Self::Output {
        NDArray::add(self, rhs)
    }
}

impl<T: Numeric> Sub<T> for &NDArray<T> {
    type Output = NDArray<T>;

    fn sub(self, rhs: T) -> Self::Output {
        NDArray::sub(self, rhs)
    }
}

impl<T: Numeric> Mul<T> for &NDArray<T> {
    type Output = NDArray<T>;

    fn mul(self, rhs: T) -> Self::Output {
        NDArray::mul(self, rhs)
    }
}

impl<T: Numeric> Div<T> for &NDArray<T> {
    type Output = NDArray<T>;

    fn div(self, rhs: T) -> Self::Output {
        NDArray::div(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{array::NDArray, AtlasNdError};

    #[test]
    fn add_uses_contiguous_fast_path_for_equal_shapes() {
        let lhs = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();
        let rhs = NDArray::from_vec(vec![2, 2], vec![5_i32, 6, 7, 8]).unwrap();

        let result = lhs.add(&rhs).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), &[6, 8, 10, 12]);
        assert!(result.is_contiguous());
    }

    #[test]
    fn add_supports_broadcasted_shapes() {
        let lhs = NDArray::from_vec(vec![2, 1], vec![1_i32, 2]).unwrap();
        let rhs = NDArray::from_vec(vec![1, 3], vec![10_i32, 20, 30]).unwrap();

        let result = lhs.add(&rhs).unwrap();

        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(result.data(), &[11, 21, 31, 12, 22, 32]);
        assert!(result.is_contiguous());
    }

    #[test]
    fn add_rejects_incompatible_shapes() {
        let lhs = NDArray::new(vec![2, 3], 1_i32);
        let rhs = NDArray::new(vec![2, 4], 1_i32);

        let error = lhs.add(&rhs).unwrap_err();

        assert_eq!(
            error,
            AtlasNdError::InvalidBroadcast {
                lhs: vec![2, 3],
                rhs: vec![2, 4],
                axis: 1,
                lhs_dim: 3,
                rhs_dim: 4,
            }
        );
    }

    #[test]
    fn scalar_ops_preserve_shape_and_layout() {
        let array = NDArray::from_vec(vec![2, 2], vec![2_i32, 4, 6, 8]).unwrap();

        assert_eq!(array.add(1).data(), &[3, 5, 7, 9]);
        assert_eq!(array.sub(1).data(), &[1, 3, 5, 7]);
        assert_eq!(array.mul(2).data(), &[4, 8, 12, 16]);
        assert_eq!(array.div(2).data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn operator_overloads_match_array_methods() {
        let lhs = NDArray::from_vec(vec![2, 1], vec![1_i32, 2]).unwrap();
        let rhs = NDArray::from_vec(vec![1, 2], vec![3_i32, 4]).unwrap();

        let method_sum = lhs.add(&rhs).unwrap();
        let method_difference = lhs.sub(&rhs).unwrap();
        let method_product = lhs.mul(&rhs).unwrap();
        let method_quotient = method_product.div(&rhs).unwrap();
        let sum = (&lhs + &rhs).unwrap();
        let difference = (&lhs - &rhs).unwrap();
        let product = (&lhs * &rhs).unwrap();
        let quotient = (&product / &rhs).unwrap();

        assert_eq!(sum.data(), method_sum.data());
        assert_eq!(difference.data(), method_difference.data());
        assert_eq!(product.data(), method_product.data());
        assert_eq!(quotient.data(), method_quotient.data());
        assert_eq!(sum.data(), &[4, 5, 5, 6]);
        assert_eq!(difference.data(), &[-2, -3, -1, -2]);
        assert_eq!(product.data(), &[3, 4, 6, 8]);
        assert_eq!(quotient.data(), &[1, 1, 2, 2]);
    }

    #[test]
    fn scalar_operator_overloads_use_infallible_path() {
        let array = NDArray::from_vec(vec![3], vec![1_i32, 2, 3]).unwrap();

        assert_eq!(array.add(1).data(), &[2, 3, 4]);
        assert_eq!(array.sub(1).data(), &[0, 1, 2]);
        assert_eq!(array.mul(2).data(), &[2, 4, 6]);
        assert_eq!(array.div(2).data(), &[0, 1, 1]);
        assert_eq!((&array + 1).data(), &[2, 3, 4]);
        assert_eq!((&array - 1).data(), &[0, 1, 2]);
        assert_eq!((&array * 2).data(), &[2, 4, 6]);
        assert_eq!((&array / 2).data(), &[0, 1, 1]);
    }
}
