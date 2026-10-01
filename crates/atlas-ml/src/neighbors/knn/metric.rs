use crate::internal::LogicalRow;

pub(crate) fn squared_distance(lhs: &[f64], rhs: &[f64]) -> f64 {
    debug_assert_eq!(lhs.len(), rhs.len());

    atlas_linalg::squared_euclidean_distance(lhs, rhs)
        .expect("KNN distance operands have matching dimensions")
}

pub(crate) fn squared_distance_to_row(lhs: LogicalRow<'_, f64>, rhs: &[f64]) -> f64 {
    debug_assert_eq!(lhs.len(), rhs.len());

    if let Some(lhs) = lhs.contiguous_slice() {
        return squared_distance(lhs, rhs);
    }

    (0..lhs.len())
        .map(|index| {
            let delta = lhs.value_at(index) - rhs[index];
            delta * delta
        })
        .sum()
}

pub(crate) fn axis_squared_distance_lower_bound(axis_delta: f64) -> f64 {
    axis_delta * axis_delta
}

pub(crate) fn ball_squared_distance_lower_bound(query: &[f64], center: &[f64], radius: f64) -> f64 {
    let center_distance = squared_distance(query, center).sqrt();
    let lower_distance = (center_distance - radius).max(0.0);

    lower_distance * lower_distance
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use super::{
        axis_squared_distance_lower_bound, ball_squared_distance_lower_bound, squared_distance,
        squared_distance_to_row,
    };
    use crate::internal::LogicalRow;

    #[test]
    fn squared_distance_handles_equal_known_and_high_dimensional_points() {
        let high_dimensional_lhs = vec![1.0_f64; 1_024];
        let high_dimensional_rhs = vec![0.0_f64; 1_024];

        assert_eq!(squared_distance(&[1.0, -2.0], &[1.0, -2.0]), 0.0);
        assert_eq!(squared_distance(&[1.0, 2.0], &[4.0, 6.0]), 25.0);
        assert_eq!(squared_distance(&high_dimensional_lhs, &high_dimensional_rhs), 1_024.0);
    }

    #[test]
    fn squared_distance_supports_logical_rows() {
        let values =
            NDArray::from_shape_vec([2, 3], vec![0.0_f64, 3.0, 1.0, 4.0, 2.0, 5.0]).unwrap();
        let transposed = values.view().transpose();

        assert_eq!(
            squared_distance_to_row(LogicalRow::from_operand(&transposed, 0), &[1.0, 4.0]),
            1.0
        );
    }

    #[test]
    fn tree_distance_bounds_are_squared_euclidean_bounds() {
        assert_eq!(axis_squared_distance_lower_bound(-3.0), 9.0);
        assert_eq!(ball_squared_distance_lower_bound(&[4.0, 0.0], &[0.0, 0.0], 1.0), 9.0);
    }
}
