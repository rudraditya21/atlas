use crate::{
    ArrayElement, ArithmeticPromote, AtlasNdResult, CastMode, NDArray, Numeric, OperandMetadata,
    RuntimeScalar,
    core::asarray::cast_array,
    layout::{broadcast::broadcast_shape, element_count},
    view::ArrayView,
};

pub enum WhereOperand<'a, T: ArrayElement> {
    Array(ArrayView<'a, T>),
    Scalar(T),
}

pub trait IntoWhereOperand<'a, T: ArrayElement> {
    fn into_where_operand(self) -> WhereOperand<'a, T>;
}

impl<'a, T: ArrayElement> IntoWhereOperand<'a, T> for &'a NDArray<T> {
    fn into_where_operand(self) -> WhereOperand<'a, T> { WhereOperand::Array(self.view()) }
}

impl<'a, T: ArrayElement> IntoWhereOperand<'a, T> for ArrayView<'a, T> {
    fn into_where_operand(self) -> WhereOperand<'a, T> { WhereOperand::Array(self) }
}

impl<'a, T: ArrayElement> IntoWhereOperand<'a, T> for &'a ArrayView<'a, T> {
    fn into_where_operand(self) -> WhereOperand<'a, T> { WhereOperand::Array(self.clone()) }
}

macro_rules! impl_scalar_where_operand {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl<'a> IntoWhereOperand<'a, $ty> for $ty {
                fn into_where_operand(self) -> WhereOperand<'a, $ty> { WhereOperand::Scalar(self) }
            }
        )+
    };
}

impl_scalar_where_operand!(bool, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);

impl NDArray<bool> {
    pub fn r#where<'x, 'y, T, U, X, Y>(
        &self,
        x: X,
        y: Y,
    ) -> AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>
    where
        T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
        U: Numeric + RuntimeScalar,
        X: IntoWhereOperand<'x, T>,
        Y: IntoWhereOperand<'y, U>,
    {
        select_where(self, x.into_where_operand(), y.into_where_operand())
    }

    pub fn where_select<'x, 'y, T, U, X, Y>(
        &self,
        x: X,
        y: Y,
    ) -> AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>
    where
        T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
        U: Numeric + RuntimeScalar,
        X: IntoWhereOperand<'x, T>,
        Y: IntoWhereOperand<'y, U>,
    {
        self.r#where(x, y)
    }
}

impl<'c> ArrayView<'c, bool> {
    pub fn r#where<'x, 'y, T, U, X, Y>(
        &self,
        x: X,
        y: Y,
    ) -> AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>
    where
        T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
        U: Numeric + RuntimeScalar,
        X: IntoWhereOperand<'x, T>,
        Y: IntoWhereOperand<'y, U>,
    {
        select_where(self, x.into_where_operand(), y.into_where_operand())
    }

    pub fn where_select<'x, 'y, T, U, X, Y>(
        &self,
        x: X,
        y: Y,
    ) -> AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>
    where
        T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
        U: Numeric + RuntimeScalar,
        X: IntoWhereOperand<'x, T>,
        Y: IntoWhereOperand<'y, U>,
    {
        self.r#where(x, y)
    }
}

fn select_where<'x, 'y, C, T, U>(
    condition: &C,
    x: WhereOperand<'x, T>,
    y: WhereOperand<'y, U>,
) -> AtlasNdResult<NDArray<<T as ArithmeticPromote<U>>::Output>>
where
    C: OperandMetadata<bool> + ?Sized,
    T: Numeric + RuntimeScalar + ArithmeticPromote<U>,
    U: Numeric + RuntimeScalar,
{
    let x = cast_where_operand::<T, <T as ArithmeticPromote<U>>::Output>(x)?;
    let y = cast_where_operand::<U, <T as ArithmeticPromote<U>>::Output>(y)?;
    let shape = broadcast_shape(condition.shape(), x.shape())?;
    let shape = broadcast_shape(&shape, y.shape())?;
    let condition_strides = crate::broadcast_strides(condition.shape(), condition.strides(), &shape)?;
    let x_strides = crate::broadcast_strides(x.shape(), x.strides(), &shape)?;
    let y_strides = crate::broadcast_strides(y.shape(), y.strides(), &shape)?;
    let mut data = Vec::with_capacity(element_count(&shape));

    for index in 0..element_count(&shape) {
        let condition_offset = offset_from_linear_index(index, condition.offset(), &shape, &condition_strides);
        let x_offset = offset_from_linear_index(index, 0, &shape, &x_strides);
        let y_offset = offset_from_linear_index(index, 0, &shape, &y_strides);
        data.push(if condition.data()[condition_offset] { x.data()[x_offset] } else { y.data()[y_offset] });
    }

    NDArray::from_row_major_parts(shape, data)
}

fn cast_where_operand<T, P>(operand: WhereOperand<'_, T>) -> AtlasNdResult<NDArray<P>>
where
    T: Numeric + RuntimeScalar,
    P: Numeric + RuntimeScalar,
{
    match operand {
        WhereOperand::Array(array) => cast_array(array.shape().to_vec(), array.iter().copied(), CastMode::Lossy),
        WhereOperand::Scalar(value) => cast_array(Vec::new(), [value], CastMode::Lossy),
    }
}

fn offset_from_linear_index(
    mut linear_index: usize,
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
) -> usize {
    let mut offset = base_offset;

    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        let coordinate = if dim == 0 { 0 } else { linear_index % dim };
        linear_index = linear_index.checked_div(dim).unwrap_or(0);
        offset += coordinate * strides[axis];
    }

    offset
}
