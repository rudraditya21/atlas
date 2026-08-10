use crate::{ArrayElement, AtlasNdResult, NDArray, OperandMetadata, view::ArrayView};

use crate::{
    internal::layout::is_contiguous_layout,
    layout::{
        broadcast::{broadcast_shape, broadcast_strides},
        element_count,
    },
};

pub enum WhereOperand<'a, T: ArrayElement> {
    Array(ArrayView<'a, T>),
    Scalar(T),
}

pub trait IntoWhereOperand<'a, T: ArrayElement> {
    fn into_where_operand(self) -> WhereOperand<'a, T>;
}

impl<'a, T: ArrayElement> IntoWhereOperand<'a, T> for &'a NDArray<T> {
    fn into_where_operand(self) -> WhereOperand<'a, T> {
        WhereOperand::Array(self.view())
    }
}

impl<'a, T: ArrayElement> IntoWhereOperand<'a, T> for ArrayView<'a, T> {
    fn into_where_operand(self) -> WhereOperand<'a, T> {
        WhereOperand::Array(self)
    }
}

impl<'a, T: ArrayElement> IntoWhereOperand<'a, T> for &'a ArrayView<'a, T> {
    fn into_where_operand(self) -> WhereOperand<'a, T> {
        WhereOperand::Array(self.clone())
    }
}

macro_rules! impl_scalar_where_operand {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl<'a> IntoWhereOperand<'a, $ty> for $ty {
                fn into_where_operand(self) -> WhereOperand<'a, $ty> {
                    WhereOperand::Scalar(self)
                }
            }
        )+
    };
}

impl_scalar_where_operand!(bool, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize, f32, f64);

impl NDArray<bool> {
    pub fn r#where<'a, T, X, Y>(&self, x: X, y: Y) -> AtlasNdResult<NDArray<T>>
    where
        T: ArrayElement,
        X: IntoWhereOperand<'a, T>,
        Y: IntoWhereOperand<'a, T>,
    {
        select_where(self, x.into_where_operand(), y.into_where_operand())
    }

    pub fn where_select<'a, T, X, Y>(&self, x: X, y: Y) -> AtlasNdResult<NDArray<T>>
    where
        T: ArrayElement,
        X: IntoWhereOperand<'a, T>,
        Y: IntoWhereOperand<'a, T>,
    {
        self.r#where(x, y)
    }
}

impl<'c> ArrayView<'c, bool> {
    pub fn r#where<'a, T, X, Y>(&self, x: X, y: Y) -> AtlasNdResult<NDArray<T>>
    where
        T: ArrayElement,
        X: IntoWhereOperand<'a, T>,
        Y: IntoWhereOperand<'a, T>,
    {
        select_where(self, x.into_where_operand(), y.into_where_operand())
    }

    pub fn where_select<'a, T, X, Y>(&self, x: X, y: Y) -> AtlasNdResult<NDArray<T>>
    where
        T: ArrayElement,
        X: IntoWhereOperand<'a, T>,
        Y: IntoWhereOperand<'a, T>,
    {
        self.r#where(x, y)
    }
}

struct BroadcastedOperand<'a, T: ArrayElement> {
    data: &'a [T],
    offset: usize,
    strides: Vec<usize>,
}

enum BroadcastedWhereOperand<'a, T: ArrayElement> {
    Array(BroadcastedOperand<'a, T>),
    Scalar(T),
}

fn select_where<'a, C, T>(
    condition: &C,
    x: WhereOperand<'a, T>,
    y: WhereOperand<'a, T>,
) -> AtlasNdResult<NDArray<T>>
where
    C: OperandMetadata<bool> + ?Sized,
    T: ArrayElement,
{
    let shape = broadcast_where_shape(condition.shape(), &x, &y)?;
    let condition = BroadcastedOperand {
        data: condition.data(),
        offset: condition.offset(),
        strides: broadcast_strides(condition.shape(), condition.strides(), &shape)?,
    };
    let x = broadcast_where_operand(x, &shape)?;
    let y = broadcast_where_operand(y, &shape)?;

    let data = if is_contiguous_layout(&shape, &condition.strides)
        && is_contiguous_or_scalar(&shape, &x)
        && is_contiguous_or_scalar(&shape, &y)
    {
        select_contiguous(&shape, &condition, &x, &y)
    } else {
        select_generic(&shape, &condition, &x, &y)
    };

    NDArray::from_row_major_parts(shape, data)
}

fn broadcast_where_shape<T: ArrayElement>(
    condition_shape: &[usize],
    x: &WhereOperand<'_, T>,
    y: &WhereOperand<'_, T>,
) -> AtlasNdResult<Vec<usize>> {
    let shape = match x {
        WhereOperand::Array(array) => broadcast_shape(condition_shape, array.shape())?,
        WhereOperand::Scalar(_) => condition_shape.to_vec(),
    };

    match y {
        WhereOperand::Array(array) => broadcast_shape(&shape, array.shape()),
        WhereOperand::Scalar(_) => Ok(shape),
    }
}

fn broadcast_where_operand<'a, T: ArrayElement>(
    operand: WhereOperand<'a, T>,
    shape: &[usize],
) -> AtlasNdResult<BroadcastedWhereOperand<'a, T>> {
    Ok(match operand {
        WhereOperand::Array(array) => BroadcastedWhereOperand::Array(BroadcastedOperand {
            data: array.data(),
            offset: array.offset(),
            strides: broadcast_strides(array.shape(), array.strides(), shape)?,
        }),
        WhereOperand::Scalar(value) => BroadcastedWhereOperand::Scalar(value),
    })
}

fn is_contiguous_or_scalar<T: ArrayElement>(
    shape: &[usize],
    operand: &BroadcastedWhereOperand<'_, T>,
) -> bool {
    match operand {
        BroadcastedWhereOperand::Array(array) => is_contiguous_layout(shape, &array.strides),
        BroadcastedWhereOperand::Scalar(_) => true,
    }
}

fn select_contiguous<T: ArrayElement>(
    shape: &[usize],
    condition: &BroadcastedOperand<'_, bool>,
    x: &BroadcastedWhereOperand<'_, T>,
    y: &BroadcastedWhereOperand<'_, T>,
) -> Vec<T> {
    let len = element_count(shape);
    let mut data = Vec::with_capacity(len);

    for index in 0..len {
        let condition_value = condition.data[condition.offset + index];
        data.push(select_value(condition_value, x, y, index, shape));
    }

    data
}

fn select_generic<T: ArrayElement>(
    shape: &[usize],
    condition: &BroadcastedOperand<'_, bool>,
    x: &BroadcastedWhereOperand<'_, T>,
    y: &BroadcastedWhereOperand<'_, T>,
) -> Vec<T> {
    let len = element_count(shape);
    let mut data = Vec::with_capacity(len);

    for index in 0..len {
        let condition_offset =
            offset_from_linear_index(index, condition.offset, shape, &condition.strides);
        let condition_value = condition.data[condition_offset];
        data.push(select_value(condition_value, x, y, index, shape));
    }

    data
}

fn select_value<T: ArrayElement>(
    condition: bool,
    x: &BroadcastedWhereOperand<'_, T>,
    y: &BroadcastedWhereOperand<'_, T>,
    linear_index: usize,
    shape: &[usize],
) -> T {
    if condition {
        operand_value(x, linear_index, shape)
    } else {
        operand_value(y, linear_index, shape)
    }
}

fn operand_value<T: ArrayElement>(
    operand: &BroadcastedWhereOperand<'_, T>,
    linear_index: usize,
    shape: &[usize],
) -> T {
    match operand {
        BroadcastedWhereOperand::Array(array) => {
            let offset =
                offset_from_linear_index(linear_index, array.offset, shape, &array.strides);
            array.data[offset]
        }
        BroadcastedWhereOperand::Scalar(value) => *value,
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
