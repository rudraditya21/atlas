use atlas_linalg::{matmul, solve};
use atlas_ndarray::NDArray;

fn generated_nonsingular_matrix(order: usize, seed: usize) -> NDArray<f64> {
    let mut values = vec![0.0; order * order];

    for row in 0..order {
        let mut off_diagonal_sum = 0.0;
        for column in 0..order {
            if row == column {
                continue;
            }

            let value = ((row * 5 + column * 3 + seed) % 7) as f64 - 3.0;
            values[row * order + column] = value;
            off_diagonal_sum += value.abs();
        }
        values[row * order + row] = off_diagonal_sum + 1.0 + (seed % 3) as f64;
    }

    NDArray::from_shape_vec([order, order], values).unwrap()
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual.iter().zip(expected).all(|(actual, expected)| (actual - expected).abs() <= 1e-10)
    );
}

#[test]
fn generated_small_systems_preserve_solve_residuals() {
    for order in 1..=5 {
        for seed in 0..8 {
            let matrix = generated_nonsingular_matrix(order, seed);
            let vector_rhs = NDArray::from_shape_vec(
                [order],
                (0..order).map(|row| ((seed + row * 2) % 11) as f64 - 5.0).collect(),
            )
            .unwrap();
            let matrix_rhs = NDArray::from_shape_vec(
                [order, 2],
                (0..order)
                    .flat_map(|row| {
                        [
                            ((seed + row * 2) % 11) as f64 - 5.0,
                            ((seed * 3 + row * 5) % 13) as f64 - 6.0,
                        ]
                    })
                    .collect(),
            )
            .unwrap();

            let vector_solution = solve(&matrix, &vector_rhs).unwrap();
            let matrix_solution = solve(&matrix, &matrix_rhs).unwrap();

            assert_close(matmul(&matrix, &vector_solution).unwrap().data(), vector_rhs.data());
            assert_close(matmul(&matrix, &matrix_solution).unwrap().data(), matrix_rhs.data());
        }
    }
}
