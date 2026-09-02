use crate::{
    ArithmeticPromote, ArrayElement, AtlasNdError, AtlasNdResult, CastMode, NDArray, Numeric,
    OperandMetadata, RuntimeScalar, ScalarValue,
    core::{asarray::cast_array, dtype::cast_scalar_to_dtype},
    internal::value_iter,
    view::ArrayView,
};

const CLIP_OP: &str = "clip";

type ClipOutput<T, L, U> = <<T as ArithmeticPromote<L>>::Output as ArithmeticPromote<U>>::Output;

impl<T> NDArray<T>
where
    T: Numeric + RuntimeScalar,
{
    /// Promotes values and bounds to a common dtype before clamping.
    pub fn clip<L, U>(&self, min: L, max: U) -> AtlasNdResult<NDArray<ClipOutput<T, L, U>>>
    where
        T: ArithmeticPromote<L>,
        L: RuntimeScalar,
        <T as ArithmeticPromote<L>>::Output: ArithmeticPromote<U>,
        U: RuntimeScalar,
        ClipOutput<T, L, U>: Numeric + RuntimeScalar + PartialOrd,
    {
        clip_operand(self, min, max)
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: Numeric + RuntimeScalar,
{
    /// Promotes values and bounds to a common dtype before clamping.
    pub fn clip<L, U>(&self, min: L, max: U) -> AtlasNdResult<NDArray<ClipOutput<T, L, U>>>
    where
        T: ArithmeticPromote<L>,
        L: RuntimeScalar,
        <T as ArithmeticPromote<L>>::Output: ArithmeticPromote<U>,
        U: RuntimeScalar,
        ClipOutput<T, L, U>: Numeric + RuntimeScalar + PartialOrd,
    {
        clip_operand(self, min, max)
    }
}

fn clip_operand<T, L, U, O>(
    operand: &O,
    min: L,
    max: U,
) -> AtlasNdResult<NDArray<ClipOutput<T, L, U>>>
where
    T: Numeric + RuntimeScalar + ArithmeticPromote<L>,
    L: RuntimeScalar,
    <T as ArithmeticPromote<L>>::Output: ArithmeticPromote<U>,
    U: RuntimeScalar,
    ClipOutput<T, L, U>: Numeric + RuntimeScalar + PartialOrd,
    O: OperandMetadata<T> + ?Sized,
{
    let min = cast_scalar_to_dtype::<L, ClipOutput<T, L, U>>(min);
    let max = cast_scalar_to_dtype::<U, ClipOutput<T, L, U>>(max);
    validate_clip_bounds(min, max)?;
    let values = cast_array::<T, ClipOutput<T, L, U>>(
        operand.shape().to_vec(),
        value_iter(operand.data(), operand.offset(), operand.shape(), operand.strides()).copied(),
        CastMode::Lossy,
    )?;
    let data = values.data().iter().copied().map(|value| clip_value(value, min, max)).collect();

    NDArray::from_row_major_parts(values.shape().to_vec(), data)
}

fn validate_clip_bounds<T>(min: T, max: T) -> AtlasNdResult<()>
where
    T: ArrayElement + RuntimeScalar + PartialOrd,
{
    if is_nan(min) || is_nan(max) {
        return Err(AtlasNdError::InvalidArgument { op: CLIP_OP, reason: "bounds must not be NaN" });
    }

    if min > max {
        return Err(AtlasNdError::InvalidArgument {
            op: CLIP_OP,
            reason: "min must be less than or equal to max",
        });
    }

    Ok(())
}

fn is_nan<T: RuntimeScalar>(value: T) -> bool {
    matches!(value.into_scalar_value(), ScalarValue::F32(value) if value.is_nan())
        || matches!(value.into_scalar_value(), ScalarValue::F64(value) if value.is_nan())
}

fn clip_value<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min { min } else if value > max { max } else { value }
}
