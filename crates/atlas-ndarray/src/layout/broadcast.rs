use crate::{AtlasNdError, AtlasNdResult, layout::stride::compute_strides};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BroadcastMetadata {
    pub shape: Vec<usize>,
    pub lhs_strides: Vec<usize>,
    pub rhs_strides: Vec<usize>,
}

pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> AtlasNdResult<Vec<usize>> {
    let ndim = lhs.len().max(rhs.len());
    let mut shape = Vec::with_capacity(ndim);

    for axis_from_end in 0..ndim {
        let lhs_dim = lhs.len().checked_sub(axis_from_end + 1).map(|index| lhs[index]).unwrap_or(1);
        let rhs_dim = rhs.len().checked_sub(axis_from_end + 1).map(|index| rhs[index]).unwrap_or(1);

        let dim = if lhs_dim == rhs_dim {
            lhs_dim
        } else if lhs_dim == 1 {
            rhs_dim
        } else if rhs_dim == 1 {
            lhs_dim
        } else {
            return Err(AtlasNdError::InvalidBroadcast {
                lhs: lhs.to_vec(),
                rhs: rhs.to_vec(),
                axis: ndim - axis_from_end - 1,
                lhs_dim,
                rhs_dim,
            });
        };

        shape.push(dim);
    }

    shape.reverse();
    Ok(shape)
}

pub fn broadcast_strides(
    shape: &[usize],
    strides: &[usize],
    target_shape: &[usize],
) -> AtlasNdResult<Vec<usize>> {
    if shape.len() != strides.len() {
        return Err(AtlasNdError::InvalidShape);
    }

    if shape.len() > target_shape.len() {
        return Err(AtlasNdError::InvalidBroadcast {
            lhs: shape.to_vec(),
            rhs: target_shape.to_vec(),
            axis: 0,
            lhs_dim: shape[0],
            rhs_dim: 1,
        });
    }

    let mut result = vec![0; target_shape.len()];
    let leading_offset = target_shape.len() - shape.len();

    for (axis, target_dim) in target_shape.iter().enumerate() {
        if axis < leading_offset {
            result[axis] = 0;
            continue;
        }

        let shape_axis = axis - leading_offset;
        let source_dim = shape[shape_axis];
        let source_stride = strides[shape_axis];

        if source_dim == *target_dim {
            result[axis] = source_stride;
        } else if source_dim == 1 {
            result[axis] = 0;
        } else {
            return Err(AtlasNdError::InvalidBroadcast {
                lhs: shape.to_vec(),
                rhs: target_shape.to_vec(),
                axis,
                lhs_dim: source_dim,
                rhs_dim: *target_dim,
            });
        }
    }

    Ok(result)
}

pub fn broadcast_pair(
    lhs_shape: &[usize],
    lhs_strides: &[usize],
    rhs_shape: &[usize],
    rhs_strides: &[usize],
) -> AtlasNdResult<BroadcastMetadata> {
    let shape = broadcast_shape(lhs_shape, rhs_shape)?;
    let lhs_strides = broadcast_strides(lhs_shape, lhs_strides, &shape)?;
    let rhs_strides = broadcast_strides(rhs_shape, rhs_strides, &shape)?;

    Ok(BroadcastMetadata { shape, lhs_strides, rhs_strides })
}

pub fn contiguous_broadcast_metadata(
    lhs: &[usize],
    rhs: &[usize],
) -> AtlasNdResult<BroadcastMetadata> {
    broadcast_pair(lhs, &compute_strides(lhs), rhs, &compute_strides(rhs))
}

#[cfg(test)]
mod tests {
    use crate::AtlasNdError;

    use super::{
        BroadcastMetadata, broadcast_pair, broadcast_shape, broadcast_strides,
        contiguous_broadcast_metadata,
    };

    #[test]
    fn broadcast_shape_aligns_trailing_dimensions() {
        assert_eq!(broadcast_shape(&[3, 1], &[2, 3, 4]).unwrap(), vec![2, 3, 4]);
        assert_eq!(broadcast_shape(&[4], &[2, 3, 4]).unwrap(), vec![2, 3, 4]);
        assert_eq!(broadcast_shape(&[], &[2, 3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[1, 5, 1], &[7, 1, 3]).unwrap(), vec![7, 5, 3]);
        assert_eq!(broadcast_shape(&[2, 1, 4], &[1, 3, 1]).unwrap(), vec![2, 3, 4]);
        assert_eq!(broadcast_shape(&[2, 0, 4], &[1, 0, 1]).unwrap(), vec![2, 0, 4]);
        assert_eq!(broadcast_shape(&[0, 4], &[1, 4]).unwrap(), vec![0, 4]);
        assert_eq!(broadcast_shape(&[1], &[0]).unwrap(), vec![0]);
    }

    #[test]
    fn broadcast_shape_supports_equal_rank_and_scalar_cases() {
        assert_eq!(broadcast_shape(&[2, 3], &[2, 3]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shape(&[], &[]).unwrap(), Vec::<usize>::new());
        assert_eq!(broadcast_shape(&[], &[0, 2, 3]).unwrap(), vec![0, 2, 3]);
        assert_eq!(broadcast_shape(&[4, 1, 1], &[]).unwrap(), vec![4, 1, 1]);
    }

    #[test]
    fn broadcast_shape_rejects_incompatible_dimensions() {
        let error = broadcast_shape(&[2, 3], &[2, 4]).unwrap_err();

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

        assert_eq!(
            broadcast_shape(&[2, 0], &[2, 3]).unwrap_err(),
            AtlasNdError::InvalidBroadcast {
                lhs: vec![2, 0],
                rhs: vec![2, 3],
                axis: 1,
                lhs_dim: 0,
                rhs_dim: 3,
            }
        );
        assert_eq!(
            broadcast_shape(&[3, 1, 2], &[4, 5, 2]).unwrap_err(),
            AtlasNdError::InvalidBroadcast {
                lhs: vec![3, 1, 2],
                rhs: vec![4, 5, 2],
                axis: 0,
                lhs_dim: 3,
                rhs_dim: 4,
            }
        );
    }

    #[test]
    fn broadcast_strides_use_zero_stride_for_expanded_dimensions() {
        assert_eq!(broadcast_strides(&[3, 1], &[1, 1], &[2, 3, 4]).unwrap(), vec![0, 1, 0]);
        assert_eq!(broadcast_strides(&[4], &[1], &[2, 3, 4]).unwrap(), vec![0, 0, 1]);
        assert_eq!(broadcast_strides(&[], &[], &[2, 3]).unwrap(), vec![0, 0]);
        assert_eq!(broadcast_strides(&[2, 1, 4], &[4, 4, 1], &[2, 3, 4]).unwrap(), vec![4, 0, 1]);
        assert_eq!(broadcast_strides(&[1, 0, 1], &[0, 1, 1], &[2, 0, 4]).unwrap(), vec![0, 1, 0]);
        assert_eq!(broadcast_strides(&[2, 3], &[3, 1], &[2, 3]).unwrap(), vec![3, 1]);
        assert_eq!(broadcast_strides(&[], &[], &[]).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn broadcast_strides_reject_invalid_shape_metadata_and_incompatible_targets() {
        assert_eq!(
            broadcast_strides(&[2, 3], &[3], &[2, 3]).unwrap_err(),
            AtlasNdError::InvalidShape
        );
        assert_eq!(
            broadcast_strides(&[2, 3, 4], &[12, 4, 1], &[3, 4]).unwrap_err(),
            AtlasNdError::InvalidBroadcast {
                lhs: vec![2, 3, 4],
                rhs: vec![3, 4],
                axis: 0,
                lhs_dim: 2,
                rhs_dim: 1,
            }
        );
        assert_eq!(
            broadcast_strides(&[2, 3], &[3, 1], &[2, 4]).unwrap_err(),
            AtlasNdError::InvalidBroadcast {
                lhs: vec![2, 3],
                rhs: vec![2, 4],
                axis: 1,
                lhs_dim: 3,
                rhs_dim: 4,
            }
        );
        assert_eq!(
            broadcast_strides(&[0, 4], &[4, 1], &[2, 4]).unwrap_err(),
            AtlasNdError::InvalidBroadcast {
                lhs: vec![0, 4],
                rhs: vec![2, 4],
                axis: 0,
                lhs_dim: 0,
                rhs_dim: 2,
            }
        );
    }

    #[test]
    fn broadcast_pair_resolves_target_shape_and_broadcasted_strides() {
        let metadata = broadcast_pair(&[2, 1, 4], &[4, 4, 1], &[3, 4], &[4, 1]).unwrap();

        assert_eq!(
            metadata,
            BroadcastMetadata {
                shape: vec![2, 3, 4],
                lhs_strides: vec![4, 0, 1],
                rhs_strides: vec![0, 4, 1],
            }
        );
    }

    #[test]
    fn broadcast_pair_handles_scalar_and_zero_sized_shapes() {
        assert_eq!(
            broadcast_pair(&[], &[], &[2, 3], &[3, 1]).unwrap(),
            BroadcastMetadata {
                shape: vec![2, 3],
                lhs_strides: vec![0, 0],
                rhs_strides: vec![3, 1],
            }
        );
        assert_eq!(
            broadcast_pair(&[2, 0, 4], &[0, 4, 1], &[1, 0, 1], &[0, 1, 1]).unwrap(),
            BroadcastMetadata {
                shape: vec![2, 0, 4],
                lhs_strides: vec![0, 4, 1],
                rhs_strides: vec![0, 1, 0],
            }
        );
    }

    #[test]
    fn contiguous_broadcast_metadata_uses_row_major_strides() {
        let metadata = contiguous_broadcast_metadata(&[2, 1, 4], &[3, 4]).unwrap();

        assert_eq!(
            metadata,
            BroadcastMetadata {
                shape: vec![2, 3, 4],
                lhs_strides: vec![4, 0, 1],
                rhs_strides: vec![0, 4, 1],
            }
        );
    }

    #[test]
    fn contiguous_broadcast_metadata_matches_row_major_rules_across_cases() {
        assert_eq!(
            contiguous_broadcast_metadata(&[1, 5, 1], &[7, 1, 3]).unwrap(),
            BroadcastMetadata {
                shape: vec![7, 5, 3],
                lhs_strides: vec![0, 1, 0],
                rhs_strides: vec![3, 0, 1],
            }
        );
        assert_eq!(
            contiguous_broadcast_metadata(&[], &[2, 3, 4]).unwrap(),
            BroadcastMetadata {
                shape: vec![2, 3, 4],
                lhs_strides: vec![0, 0, 0],
                rhs_strides: vec![12, 4, 1],
            }
        );
    }
}
