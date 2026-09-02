mod broadcast;
mod contiguous;
mod dispatch;
mod minmax;
mod scalar;
mod strided;

use std::ops::{Add, Div, Mul, Rem, Sub};

use crate::core::dtype::ArithmeticPromote;
use crate::{AtlasNdResult, NDArray, Numeric, RuntimeScalar, core::dtype};

use self::{
    contiguous::{elementwise_add_contiguous, elementwise_mul_contiguous},
    dispatch::{BinaryOperand, dispatch_elementwise_binary, dispatch_elementwise_binary_with},
    scalar::{add_scalar_lhs, add_scalar_rhs, mul_scalar_lhs, mul_scalar_rhs},
};

pub use self::minmax::ElementwiseMinMax;

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

pub trait RemOperand<T: Numeric> {
    type Output;

    fn rem_into(self, lhs: &NDArray<T>) -> Self::Output;
}

pub trait MinOperand<T: Numeric> {
    type Output;

    fn minimum_with(self, lhs: &NDArray<T>) -> Self::Output;
}

pub trait MaxOperand<T: Numeric> {
    type Output;

    fn maximum_with(self, lhs: &NDArray<T>) -> Self::Output;
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

    pub fn rem<Rhs>(&self, rhs: Rhs) -> Rhs::Output
    where
        Rhs: RemOperand<T>,
    {
        rhs.rem_into(self)
    }

    /// Returns the elementwise minimum; a single floating NaN selects the numeric operand.
    pub fn minimum<Rhs>(&self, rhs: Rhs) -> Rhs::Output
    where
        Rhs: MinOperand<T>,
    {
        rhs.minimum_with(self)
    }

    /// Returns the elementwise maximum; a single floating NaN selects the numeric operand.
    pub fn maximum<Rhs>(&self, rhs: Rhs) -> Rhs::Output
    where
        Rhs: MaxOperand<T>,
    {
        rhs.maximum_with(self)
    }

    pub fn add_scalar(&self, scalar: T) -> Self {
        dispatch_elementwise_binary_with(
            self,
            BinaryOperand::Scalar(scalar),
            add_scalar_rhs,
            add_scalar_lhs,
            elementwise_add_contiguous,
            |lhs, rhs| lhs + rhs,
        )
        .expect("scalar rhs dispatch must not fail")
    }

    pub fn sub_scalar(&self, scalar: T) -> Self {
        dispatch_elementwise_binary(self, BinaryOperand::Scalar(scalar), |lhs, rhs| lhs - rhs)
            .expect("scalar rhs dispatch must not fail")
    }

    pub fn mul_scalar(&self, scalar: T) -> Self {
        dispatch_elementwise_binary_with(
            self,
            BinaryOperand::Scalar(scalar),
            mul_scalar_rhs,
            mul_scalar_lhs,
            elementwise_mul_contiguous,
            |lhs, rhs| lhs * rhs,
        )
        .expect("scalar rhs dispatch must not fail")
    }

    pub fn div_scalar(&self, scalar: T) -> Self {
        dispatch_elementwise_binary(self, BinaryOperand::Scalar(scalar), |lhs, rhs| lhs / rhs)
            .expect("scalar rhs dispatch must not fail")
    }

    pub fn rem_scalar(&self, scalar: T) -> Self {
        dispatch_elementwise_binary(self, BinaryOperand::Scalar(scalar), |lhs, rhs| lhs % rhs)
            .expect("scalar rhs dispatch must not fail")
    }

    fn minimum_scalar(&self, scalar: T) -> Self
    where
        T: ElementwiseMinMax,
    {
        dispatch_elementwise_binary(self, BinaryOperand::Scalar(scalar), |lhs, rhs| {
            lhs.elementwise_min(rhs)
        })
        .expect("scalar rhs dispatch must not fail")
    }

    fn maximum_scalar(&self, scalar: T) -> Self
    where
        T: ElementwiseMinMax,
    {
        dispatch_elementwise_binary(self, BinaryOperand::Scalar(scalar), |lhs, rhs| {
            lhs.elementwise_max(rhs)
        })
        .expect("scalar rhs dispatch must not fail")
    }

    fn add_array(&self, rhs: &Self) -> AtlasNdResult<Self> {
        dispatch_elementwise_binary_with(
            self,
            BinaryOperand::Array(rhs),
            add_scalar_rhs,
            add_scalar_lhs,
            elementwise_add_contiguous,
            |lhs, rhs| lhs + rhs,
        )
    }

    fn mul_array(&self, rhs: &Self) -> AtlasNdResult<Self> {
        dispatch_elementwise_binary_with(
            self,
            BinaryOperand::Array(rhs),
            mul_scalar_rhs,
            mul_scalar_lhs,
            elementwise_mul_contiguous,
            |lhs, rhs| lhs * rhs,
        )
    }

    fn rem_array(&self, rhs: &Self) -> AtlasNdResult<Self> {
        dispatch_elementwise_binary(self, BinaryOperand::Array(rhs), |lhs, rhs| lhs % rhs)
    }

    fn minimum_array(&self, rhs: &Self) -> AtlasNdResult<Self>
    where
        T: ElementwiseMinMax,
    {
        dispatch_elementwise_binary(self, BinaryOperand::Array(rhs), |lhs, rhs| {
            lhs.elementwise_min(rhs)
        })
    }

    fn maximum_array(&self, rhs: &Self) -> AtlasNdResult<Self>
    where
        T: ElementwiseMinMax,
    {
        dispatch_elementwise_binary(self, BinaryOperand::Array(rhs), |lhs, rhs| {
            lhs.elementwise_max(rhs)
        })
    }
}

pub(super) fn from_owned_parts<T: Numeric>(shape: Vec<usize>, data: Vec<T>) -> NDArray<T> {
    NDArray::from_row_major_parts(shape, data)
        .expect("internal owned array construction must preserve row-major ndarray invariants")
}

fn promoted_array_array<T, U, P, F>(
    lhs: &NDArray<T>,
    rhs: &NDArray<U>,
    op: F,
) -> AtlasNdResult<NDArray<P>>
where
    T: Numeric + RuntimeScalar,
    U: Numeric + RuntimeScalar,
    P: Numeric + RuntimeScalar,
    F: Fn(&NDArray<P>, &NDArray<P>) -> AtlasNdResult<NDArray<P>>,
{
    let lhs = dtype::cast_array_for_promotion::<T, P>(lhs);
    let rhs = dtype::cast_array_for_promotion::<U, P>(rhs);
    op(&lhs, &rhs)
}

fn promoted_array_scalar<T, U, P, F>(lhs: &NDArray<T>, rhs: U, op: F) -> NDArray<P>
where
    T: Numeric + RuntimeScalar,
    U: dtype::ArithmeticScalar,
    P: Numeric + RuntimeScalar,
    F: Fn(&NDArray<P>, P) -> NDArray<P>,
{
    let lhs = dtype::cast_array_for_promotion::<T, P>(lhs);
    let rhs = dtype::cast_scalar_for_promotion::<U, P>(rhs);
    op(&lhs, rhs)
}

impl<T, U> AddOperand<T> for &NDArray<U>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: Numeric + RuntimeScalar,
{
    type Output = AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>;

    fn add_to(self, lhs: &NDArray<T>) -> Self::Output {
        promoted_array_array::<T, U, <T as ArithmeticPromote<U>>::Output, _>(
            lhs,
            self,
            |lhs, rhs| lhs.add_array(rhs),
        )
    }
}

impl<T, U> AddOperand<T> for U
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: dtype::ArithmeticScalar,
{
    type Output = NDArray<<T as ArithmeticPromote<U>>::Output>;

    fn add_to(self, lhs: &NDArray<T>) -> Self::Output {
        promoted_array_scalar::<T, U, <T as ArithmeticPromote<U>>::Output, _>(
            lhs,
            self,
            |lhs, rhs| lhs.add_scalar(rhs),
        )
    }
}

impl<T, U> SubOperand<T> for &NDArray<U>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: Numeric + RuntimeScalar,
{
    type Output = AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>;

    fn sub_from(self, lhs: &NDArray<T>) -> Self::Output {
        promoted_array_array::<T, U, <T as ArithmeticPromote<U>>::Output, _>(
            lhs,
            self,
            |lhs, rhs| {
                dispatch_elementwise_binary(lhs, BinaryOperand::Array(rhs), |lhs, rhs| lhs - rhs)
            },
        )
    }
}

impl<T, U> SubOperand<T> for U
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: dtype::ArithmeticScalar,
{
    type Output = NDArray<<T as ArithmeticPromote<U>>::Output>;

    fn sub_from(self, lhs: &NDArray<T>) -> Self::Output {
        promoted_array_scalar::<T, U, <T as ArithmeticPromote<U>>::Output, _>(
            lhs,
            self,
            |lhs, rhs| lhs.sub_scalar(rhs),
        )
    }
}

impl<T, U> MulOperand<T> for &NDArray<U>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: Numeric + RuntimeScalar,
{
    type Output = AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>;

    fn mul_with(self, lhs: &NDArray<T>) -> Self::Output {
        promoted_array_array::<T, U, <T as ArithmeticPromote<U>>::Output, _>(
            lhs,
            self,
            |lhs, rhs| lhs.mul_array(rhs),
        )
    }
}

impl<T, U> MulOperand<T> for U
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: dtype::ArithmeticScalar,
{
    type Output = NDArray<<T as ArithmeticPromote<U>>::Output>;

    fn mul_with(self, lhs: &NDArray<T>) -> Self::Output {
        promoted_array_scalar::<T, U, <T as ArithmeticPromote<U>>::Output, _>(
            lhs,
            self,
            |lhs, rhs| lhs.mul_scalar(rhs),
        )
    }
}

impl<T, U> DivOperand<T> for &NDArray<U>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: Numeric + RuntimeScalar,
{
    type Output = AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>;

    fn div_into(self, lhs: &NDArray<T>) -> Self::Output {
        promoted_array_array::<T, U, <T as ArithmeticPromote<U>>::Output, _>(
            lhs,
            self,
            |lhs, rhs| {
                dispatch_elementwise_binary(lhs, BinaryOperand::Array(rhs), |lhs, rhs| lhs / rhs)
            },
        )
    }
}

impl<T, U> DivOperand<T> for U
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: dtype::ArithmeticScalar,
{
    type Output = NDArray<<T as ArithmeticPromote<U>>::Output>;

    fn div_into(self, lhs: &NDArray<T>) -> Self::Output {
        promoted_array_scalar::<T, U, <T as ArithmeticPromote<U>>::Output, _>(
            lhs,
            self,
            |lhs, rhs| lhs.div_scalar(rhs),
        )
    }
}

impl<T, U> RemOperand<T> for &NDArray<U>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: Numeric + RuntimeScalar,
{
    type Output = AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>;

    fn rem_into(self, lhs: &NDArray<T>) -> Self::Output {
        promoted_array_array::<T, U, <T as ArithmeticPromote<U>>::Output, _>(
            lhs,
            self,
            |lhs, rhs| lhs.rem_array(rhs),
        )
    }
}

impl<T, U> RemOperand<T> for U
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: dtype::ArithmeticScalar,
{
    type Output = NDArray<<T as ArithmeticPromote<U>>::Output>;

    fn rem_into(self, lhs: &NDArray<T>) -> Self::Output {
        promoted_array_scalar::<T, U, <T as ArithmeticPromote<U>>::Output, _>(
            lhs,
            self,
            |lhs, rhs| lhs.rem_scalar(rhs),
        )
    }
}

impl<T, U> MinOperand<T> for &NDArray<U>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: Numeric + RuntimeScalar,
    <T as ArithmeticPromote<U>>::Output: ElementwiseMinMax,
{
    type Output = AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>;

    fn minimum_with(self, lhs: &NDArray<T>) -> Self::Output {
        promoted_array_array::<T, U, <T as ArithmeticPromote<U>>::Output, _>(
            lhs,
            self,
            |lhs, rhs| lhs.minimum_array(rhs),
        )
    }
}

impl<T, U> MinOperand<T> for U
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: dtype::ArithmeticScalar,
    <T as ArithmeticPromote<U>>::Output: ElementwiseMinMax,
{
    type Output = NDArray<<T as ArithmeticPromote<U>>::Output>;

    fn minimum_with(self, lhs: &NDArray<T>) -> Self::Output {
        promoted_array_scalar::<T, U, <T as ArithmeticPromote<U>>::Output, _>(
            lhs,
            self,
            |lhs, rhs| lhs.minimum_scalar(rhs),
        )
    }
}

impl<T, U> MaxOperand<T> for &NDArray<U>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: Numeric + RuntimeScalar,
    <T as ArithmeticPromote<U>>::Output: ElementwiseMinMax,
{
    type Output = AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>;

    fn maximum_with(self, lhs: &NDArray<T>) -> Self::Output {
        promoted_array_array::<T, U, <T as ArithmeticPromote<U>>::Output, _>(
            lhs,
            self,
            |lhs, rhs| lhs.maximum_array(rhs),
        )
    }
}

impl<T, U> MaxOperand<T> for U
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: dtype::ArithmeticScalar,
    <T as ArithmeticPromote<U>>::Output: ElementwiseMinMax,
{
    type Output = NDArray<<T as ArithmeticPromote<U>>::Output>;

    fn maximum_with(self, lhs: &NDArray<T>) -> Self::Output {
        promoted_array_scalar::<T, U, <T as ArithmeticPromote<U>>::Output, _>(
            lhs,
            self,
            |lhs, rhs| lhs.maximum_scalar(rhs),
        )
    }
}

impl<T, U> Add<&NDArray<U>> for &NDArray<T>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: Numeric + RuntimeScalar,
{
    type Output = AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>;

    fn add(self, rhs: &NDArray<U>) -> Self::Output {
        NDArray::add(self, rhs)
    }
}

impl<T, U> Sub<&NDArray<U>> for &NDArray<T>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: Numeric + RuntimeScalar,
{
    type Output = AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>;

    fn sub(self, rhs: &NDArray<U>) -> Self::Output {
        NDArray::sub(self, rhs)
    }
}

impl<T, U> Mul<&NDArray<U>> for &NDArray<T>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: Numeric + RuntimeScalar,
{
    type Output = AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>;

    fn mul(self, rhs: &NDArray<U>) -> Self::Output {
        NDArray::mul(self, rhs)
    }
}

impl<T, U> Div<&NDArray<U>> for &NDArray<T>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: Numeric + RuntimeScalar,
{
    type Output = AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>;

    fn div(self, rhs: &NDArray<U>) -> Self::Output {
        NDArray::div(self, rhs)
    }
}

impl<T, U> Rem<&NDArray<U>> for &NDArray<T>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: Numeric + RuntimeScalar,
{
    type Output = AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>;

    fn rem(self, rhs: &NDArray<U>) -> Self::Output {
        NDArray::rem(self, rhs)
    }
}

impl<T, U> Add<U> for &NDArray<T>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: dtype::ArithmeticScalar,
{
    type Output = <U as AddOperand<T>>::Output;

    fn add(self, rhs: U) -> Self::Output {
        NDArray::add(self, rhs)
    }
}

impl<T, U> Sub<U> for &NDArray<T>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: dtype::ArithmeticScalar,
{
    type Output = <U as SubOperand<T>>::Output;

    fn sub(self, rhs: U) -> Self::Output {
        NDArray::sub(self, rhs)
    }
}

impl<T, U> Mul<U> for &NDArray<T>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: dtype::ArithmeticScalar,
{
    type Output = <U as MulOperand<T>>::Output;

    fn mul(self, rhs: U) -> Self::Output {
        NDArray::mul(self, rhs)
    }
}

impl<T, U> Div<U> for &NDArray<T>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: dtype::ArithmeticScalar,
{
    type Output = <U as DivOperand<T>>::Output;

    fn div(self, rhs: U) -> Self::Output {
        NDArray::div(self, rhs)
    }
}

impl<T, U> Rem<U> for &NDArray<T>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: dtype::ArithmeticScalar,
{
    type Output = <U as RemOperand<T>>::Output;

    fn rem(self, rhs: U) -> Self::Output {
        NDArray::rem(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray};

    fn assert_array_eq<T>(lhs: &NDArray<T>, rhs: &NDArray<T>)
    where
        T: crate::Numeric + PartialEq,
    {
        assert_eq!(lhs.shape(), rhs.shape());
        assert_eq!(lhs.strides(), rhs.strides());
        assert_eq!(lhs.data(), rhs.data());
        assert_eq!(lhs.is_contiguous(), rhs.is_contiguous());
    }

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
        let lhs = NDArray::new(vec![2, 3], 1_i32).unwrap();
        let rhs = NDArray::new(vec![2, 4], 1_i32).unwrap();

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

    #[test]
    fn scalar_rhs_values_match_scalar_shaped_array_semantics() {
        let array = NDArray::from_vec([2, 2], vec![2_i32, 4, 6, 8]).unwrap();
        let scalar = NDArray::from_shape_vec([], vec![2_i32]).unwrap();

        assert_array_eq(&array.add(2), &array.add(&scalar).unwrap());
        assert_array_eq(&array.sub(2), &array.sub(&scalar).unwrap());
        assert_array_eq(&array.mul(2), &array.mul(&scalar).unwrap());
        assert_array_eq(&array.div(2), &array.div(&scalar).unwrap());
    }

    #[test]
    fn scalar_paths_match_broadcast_array_paths_for_scalar_and_empty_outputs() {
        let scalar_array = NDArray::from_shape_vec([], vec![9_i32]).unwrap();
        let scalar_rhs = NDArray::from_shape_vec([], vec![3_i32]).unwrap();
        let empty_array = NDArray::<i32>::zeros([0, 3]).unwrap();
        let empty_rhs = NDArray::from_shape_vec([], vec![5_i32]).unwrap();

        assert_array_eq(&scalar_array.add(3), &scalar_array.add(&scalar_rhs).unwrap());
        assert_array_eq(&scalar_array.sub(3), &scalar_array.sub(&scalar_rhs).unwrap());
        assert_array_eq(&empty_array.mul(5), &empty_array.mul(&empty_rhs).unwrap());
        assert_array_eq(&empty_array.div(5), &empty_array.div(&empty_rhs).unwrap());
    }

    #[test]
    fn broadcast_arithmetic_supports_scalar_fast_path_in_both_operand_orders() {
        let array = NDArray::from_vec([2, 2], vec![8_i32, 10, 12, 14]).unwrap();
        let scalar = NDArray::from_shape_vec([], vec![2_i32]).unwrap();

        assert_eq!(array.sub(&scalar).unwrap().data(), &[6, 8, 10, 12]);
        assert_eq!(scalar.sub(&array).unwrap().data(), &[-6, -8, -10, -12]);
        assert_eq!(array.div(&scalar).unwrap().data(), &[4, 5, 6, 7]);
    }

    #[test]
    fn broadcast_arithmetic_supports_row_fast_path_in_both_operand_orders() {
        let matrix = NDArray::from_vec([2, 3], vec![10_i32, 20, 30, 40, 50, 60]).unwrap();
        let row = NDArray::from_vec([1, 3], vec![1_i32, 2, 3]).unwrap();

        assert_eq!(matrix.sub(&row).unwrap().data(), &[9, 18, 27, 39, 48, 57]);
        assert_eq!(row.sub(&matrix).unwrap().data(), &[-9, -18, -27, -39, -48, -57]);
        assert_eq!(matrix.mul(&row).unwrap().data(), &[10, 40, 90, 40, 100, 180]);
    }

    #[test]
    fn broadcast_arithmetic_supports_column_fast_path_in_both_operand_orders() {
        let matrix = NDArray::from_vec([2, 3], vec![10_i32, 20, 30, 40, 50, 60]).unwrap();
        let column = NDArray::from_vec([2, 1], vec![1_i32, 2]).unwrap();

        assert_eq!(matrix.sub(&column).unwrap().data(), &[9, 19, 29, 38, 48, 58]);
        assert_eq!(column.sub(&matrix).unwrap().data(), &[-9, -19, -29, -38, -48, -58]);
        assert_eq!(matrix.add(&column).unwrap().data(), &[11, 21, 31, 42, 52, 62]);
    }
}
