use super::{array::NDArray, traits::Numeric};
use std::ops::{Add, Div, Mul, Sub};

impl<T: Numeric> NDArray<T> {
    fn elementwise<F>(&self, rhs: &Self, op: F) -> Self
    where
        F: Fn(T, T) -> T,
    {
        assert_eq!(
            self.shape, rhs.shape,
            "Shape mismatch {:?} vs {:?}",
            self.shape, rhs.shape
        );

        let data = self
            .data
            .iter()
            .copied()
            .zip(rhs.data.iter().copied())
            .map(|(a, b)| op(a, b))
            .collect();

        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }

    fn scalar<F>(&self, scalar: T, op: F) -> Self
    where
        F: Fn(T, T) -> T,
    {
        let data = self.data.iter().copied().map(|x| op(x, scalar)).collect();

        Self {
            data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
        }
    }
}

impl<T: Numeric> Add for &NDArray<T> {
    type Output = NDArray<T>;
    fn add(self, rhs: Self) -> Self::Output {
        self.elementwise(rhs, |a, b| a + b)
    }
}

impl<T: Numeric> Sub for &NDArray<T> {
    type Output = NDArray<T>;
    fn sub(self, rhs: Self) -> Self::Output {
        self.elementwise(rhs, |a, b| a - b)
    }
}

impl<T: Numeric> Mul for &NDArray<T> {
    type Output = NDArray<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        self.elementwise(rhs, |a, b| a * b)
    }
}

impl<T: Numeric> Div for &NDArray<T> {
    type Output = NDArray<T>;
    fn div(self, rhs: Self) -> Self::Output {
        self.elementwise(rhs, |a, b| a / b)
    }
}

impl<T: Numeric> Add<T> for &NDArray<T> {
    type Output = NDArray<T>;
    fn add(self, rhs: T) -> Self::Output {
        self.scalar(rhs, |a, b| a + b)
    }
}

impl<T: Numeric> Sub<T> for &NDArray<T> {
    type Output = NDArray<T>;

    fn sub(self, rhs: T) -> Self::Output {
        self.scalar(rhs, |a, b| a - b)
    }
}

impl<T: Numeric> Mul<T> for &NDArray<T> {
    type Output = NDArray<T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.scalar(rhs, |a, b| a * b)
    }
}

impl<T: Numeric> Div<T> for &NDArray<T> {
    type Output = NDArray<T>;

    fn div(self, rhs: T) -> Self::Output {
        self.scalar(rhs, |a, b| a / b)
    }
}
