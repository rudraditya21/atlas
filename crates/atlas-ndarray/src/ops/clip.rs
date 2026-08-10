use crate::{ArrayElement, AtlasNdError, AtlasNdResult, NDArray, OperandMetadata, view::ArrayView};

use crate::internal::{layout::is_contiguous_layout, value_iter};

const CLIP_OP: &str = "clip";

impl<T> NDArray<T>
where
    T: ArrayElement + PartialOrd,
{
    pub fn clip(&self, min: T, max: T) -> AtlasNdResult<Self> {
        clip_operand(self, min, max)
    }
}

impl<'a, T> ArrayView<'a, T>
where
    T: ArrayElement + PartialOrd,
{
    pub fn clip(&self, min: T, max: T) -> AtlasNdResult<NDArray<T>> {
        clip_operand(self, min, max)
    }
}

fn clip_operand<T, O>(operand: &O, min: T, max: T) -> AtlasNdResult<NDArray<T>>
where
    T: ArrayElement + PartialOrd,
    O: OperandMetadata<T> + ?Sized,
{
    validate_clip_bounds(min, max)?;

    let data = if is_contiguous_layout(operand.shape(), operand.strides()) {
        operand
            .dense_slice()
            .expect("row-major contiguous operands always expose a dense slice")
            .iter()
            .copied()
            .map(|value| clip_value(value, min, max))
            .collect()
    } else {
        value_iter(operand.data(), operand.offset(), operand.shape(), operand.strides())
            .copied()
            .map(|value| clip_value(value, min, max))
            .collect()
    };

    NDArray::from_row_major_parts(operand.shape().to_vec(), data)
}

fn validate_clip_bounds<T>(min: T, max: T) -> AtlasNdResult<()>
where
    T: ArrayElement + PartialOrd,
{
    if min > max {
        return Err(AtlasNdError::InvalidArgument {
            op: CLIP_OP,
            reason: "min must be less than or equal to max",
        });
    }

    Ok(())
}

fn clip_value<T>(value: T, min: T, max: T) -> T
where
    T: ArrayElement + PartialOrd,
{
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray};

    #[test]
    fn clip_clamps_contiguous_arrays_in_place_order() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let clipped = array.clip(1, 4).unwrap();

        assert_eq!(clipped.shape(), &[2, 3]);
        assert_eq!(clipped.strides(), &[3, 1]);
        assert_eq!(clipped.data(), &[1, 1, 2, 3, 4, 4]);
    }

    #[test]
    fn clip_materializes_strided_views_in_logical_order() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().transpose();
        let clipped = view.clip(1, 4).unwrap();

        assert_eq!(clipped.shape(), &[3, 2]);
        assert_eq!(clipped.strides(), &[2, 1]);
        assert_eq!(clipped.data(), &[1, 3, 1, 4, 2, 4]);
    }

    #[test]
    fn clip_preserves_scalar_and_empty_shapes() {
        let scalar = NDArray::from_shape_vec([], vec![7_i32]).unwrap();
        let empty = NDArray::<i32>::from_shape_vec([2, 0, 3], Vec::new()).unwrap();

        let clipped_scalar = scalar.clip(0, 5).unwrap();
        let clipped_empty = empty.clip(0, 5).unwrap();

        assert_eq!(clipped_scalar.shape(), &[] as &[usize]);
        assert_eq!(clipped_scalar.data(), &[5]);
        assert_eq!(clipped_empty.shape(), &[2, 0, 3]);
        assert!(clipped_empty.data().is_empty());
    }

    #[test]
    fn clip_rejects_inverted_bounds() {
        let array = NDArray::from_shape_vec([2], vec![1_i32, 2]).unwrap();

        assert_eq!(
            array.clip(3, 2).unwrap_err(),
            AtlasNdError::InvalidArgument {
                op: "clip",
                reason: "min must be less than or equal to max",
            }
        );
    }
}
