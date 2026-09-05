use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, AxisIndex, NDArray,
    core::axis::{normalize_axis, normalize_scalar_index},
};

impl<T: ArrayElement> NDArray<T> {
    /// Scatters `values` along `axis` at `indices`.
    ///
    /// `values` must have this array's shape with the selected axis replaced by
    /// `indices.len()`. Negative indices address from the end of the axis. Duplicate indices are
    /// applied from left to right, so the last corresponding value wins.
    pub fn put<I: AxisIndex, A: AxisIndex>(
        &mut self,
        indices: &[I],
        values: &NDArray<T>,
        axis: A,
    ) -> AtlasNdResult<()> {
        let axis = normalize_axis(axis, self.ndim())?;
        let selected = indices
            .iter()
            .map(|index| normalize_scalar_index(*index, axis, self.shape[axis]))
            .collect::<AtlasNdResult<Vec<_>>>()?;
        let mut expected_shape = self.shape.clone();
        expected_shape[axis] = selected.len();
        if values.shape() != expected_shape {
            return Err(AtlasNdError::InvalidArgument {
                op: "put",
                reason: "values shape must match gathered shape",
            });
        }

        for (value_index, value) in values.data().iter().enumerate() {
            let mut remainder = value_index;
            let mut offset = 0;
            for dimension in (0..expected_shape.len()).rev() {
                let coordinate = remainder % expected_shape[dimension];
                remainder /= expected_shape[dimension];
                let target_coordinate =
                    if dimension == axis { selected[coordinate] } else { coordinate };
                offset += target_coordinate * self.strides[dimension];
            }
            self.data[offset] = *value;
        }
        Ok(())
    }

    /// Alias for [`NDArray::put`].
    pub fn scatter<I: AxisIndex, A: AxisIndex>(
        &mut self,
        indices: &[I],
        values: &NDArray<T>,
        axis: A,
    ) -> AtlasNdResult<()> {
        self.put(indices, values, axis)
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray};

    #[test]
    fn put_scatters_values_along_an_axis() {
        let mut array = NDArray::from_shape_vec([2, 3], vec![0_i32; 6]).unwrap();
        let values = NDArray::from_shape_vec([2, 2], vec![1, 2, 3, 4]).unwrap();

        array.put(&[2, 0], &values, 1_i32).unwrap();

        assert_eq!(array.data(), &[2, 0, 1, 4, 0, 3]);
    }

    #[test]
    fn put_uses_last_write_for_duplicate_indices() {
        let mut array = NDArray::from_shape_vec([3], vec![0_i32; 3]).unwrap();
        let values = NDArray::from_shape_vec([3], vec![1, 2, 3]).unwrap();

        array.scatter(&[1, 1, -1], &values, 0_i32).unwrap();

        assert_eq!(array.data(), &[0, 2, 3]);
    }

    #[test]
    fn put_validates_value_shape_and_indices() {
        let mut array = NDArray::from_shape_vec([2, 3], vec![0_i32; 6]).unwrap();
        let invalid_shape = NDArray::from_shape_vec([2, 1], vec![1_i32; 2]).unwrap();

        assert_eq!(
            array.put(&[0, 1], &invalid_shape, 1_i32).unwrap_err(),
            AtlasNdError::InvalidArgument {
                op: "put",
                reason: "values shape must match gathered shape",
            }
        );
        assert_eq!(
            array.put(&[3], &invalid_shape, 1_i32).unwrap_err(),
            AtlasNdError::IndexOutOfBounds { axis: 1, index: 3, dim: 3 }
        );
    }
}
