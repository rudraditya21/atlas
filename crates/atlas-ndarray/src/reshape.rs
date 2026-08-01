use super::{
    error::{AtlasNdError, AtlasNdResult},
    stride::element_count,
    traits::Numeric,
    view::ArrayView,
};

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn reshape(mut self, new_shape: Vec<usize>) -> AtlasNdResult<Self> {
        let old_size = element_count(&self.shape);
        let new_size = element_count(&new_shape);

        if old_size != new_size {
            return Err(AtlasNdError::InvalidReshape {
                from: self.shape.clone(),
                to: new_shape,
                reason: "element count must remain unchanged",
            });
        }

        if !self.is_contiguous() {
            return Err(AtlasNdError::InvalidReshape {
                from: self.shape.clone(),
                to: new_shape,
                reason: "only contiguous views can be reshaped",
            });
        }

        self.shape = new_shape;
        self.strides = super::stride::compute_strides(&self.shape);
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use crate::{array::NDArray, error::AtlasNdError};

    #[test]
    fn reshape_allows_contiguous_views() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let reshaped = array.view().reshape(vec![3, 2]).unwrap();

        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.strides(), &[2, 1]);
        assert_eq!(*reshaped.get(&[2, 1]).unwrap(), 5);
    }

    #[test]
    fn reshape_rejects_non_contiguous_views() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().slice(&[0, 1], vec![2, 2]).unwrap();
        let error = view.reshape(vec![4]).unwrap_err();

        assert_eq!(
            error,
            AtlasNdError::InvalidReshape {
                from: vec![2, 2],
                to: vec![4],
                reason: "only contiguous views can be reshaped",
            }
        );
    }

    #[test]
    fn reshape_rejects_different_element_counts() {
        let array = NDArray::new(vec![2, 3], 0_i32);
        let error = array.view().reshape(vec![5]).unwrap_err();

        assert_eq!(
            error,
            AtlasNdError::InvalidReshape {
                from: vec![2, 3],
                to: vec![5],
                reason: "element count must remain unchanged",
            }
        );
    }
}
