use num_traits::{Num, NumAssign};

pub trait Numeric:
    Num + NumAssign + Copy + Clone + Send + Sync + std::fmt::Debug + 'static
{
}

impl<T> Numeric for T where
    T: Num + NumAssign + Copy + Clone + Send + Sync + std::fmt::Debug + 'static
{
}

pub trait ShapeArg {
    fn into_shape_vec(self) -> Vec<usize>;
}

impl ShapeArg for Vec<usize> {
    fn into_shape_vec(self) -> Vec<usize> {
        self
    }
}

impl ShapeArg for &[usize] {
    fn into_shape_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl ShapeArg for &Vec<usize> {
    fn into_shape_vec(self) -> Vec<usize> {
        self.clone()
    }
}

impl<const N: usize> ShapeArg for [usize; N] {
    fn into_shape_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> ShapeArg for &[usize; N] {
    fn into_shape_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}
