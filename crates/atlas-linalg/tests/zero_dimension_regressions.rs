use atlas_linalg::{DotOutput, batched_dot, dot, matmul};
use atlas_ndarray::NDArray;

#[test]
fn matrix_products_preserve_zero_row_column_and_inner_dimensions() {
    let zero_rows =
        matmul(&NDArray::<i32>::zeros([0, 3]).unwrap(), &NDArray::<i32>::zeros([3, 2]).unwrap())
            .unwrap();
    let zero_columns =
        matmul(&NDArray::<i32>::zeros([2, 3]).unwrap(), &NDArray::<i32>::zeros([3, 0]).unwrap())
            .unwrap();
    let zero_inner =
        matmul(&NDArray::<i32>::zeros([2, 0]).unwrap(), &NDArray::<i32>::zeros([0, 3]).unwrap())
            .unwrap();

    assert_eq!(zero_rows.shape(), &[0, 2]);
    assert!(zero_rows.data().is_empty());
    assert_eq!(zero_columns.shape(), &[2, 0]);
    assert!(zero_columns.data().is_empty());
    assert_eq!(zero_inner.shape(), &[2, 3]);
    assert_eq!(zero_inner.data(), &[0; 6]);
}

#[test]
fn vector_and_batched_operations_preserve_zero_lengths_and_batches() {
    let vector_matrix =
        matmul(&NDArray::<i32>::zeros([0]).unwrap(), &NDArray::<i32>::zeros([0, 3]).unwrap())
            .unwrap();
    let matrix_vector =
        matmul(&NDArray::<i32>::zeros([3, 0]).unwrap(), &NDArray::<i32>::zeros([0]).unwrap())
            .unwrap();
    let dot_product =
        dot(&NDArray::<i32>::zeros([0]).unwrap(), &NDArray::<i32>::zeros([0]).unwrap()).unwrap();
    let batched_product = matmul(
        &NDArray::<i32>::zeros([0, 2, 0]).unwrap(),
        &NDArray::<i32>::zeros([0, 0, 3]).unwrap(),
    )
    .unwrap();
    let batched_vectors = batched_dot(
        &NDArray::<i32>::zeros([2, 0]).unwrap(),
        &NDArray::<i32>::zeros([2, 0]).unwrap(),
    )
    .unwrap();

    assert_eq!(vector_matrix.shape(), &[3]);
    assert_eq!(vector_matrix.data(), &[0; 3]);
    assert_eq!(matrix_vector.shape(), &[3]);
    assert_eq!(matrix_vector.data(), &[0; 3]);
    assert_eq!(scalar_dot(dot_product), 0);
    assert_eq!(batched_product.shape(), &[0, 2, 3]);
    assert!(batched_product.data().is_empty());
    assert_eq!(batched_vectors.shape(), &[2]);
    assert_eq!(batched_vectors.data(), &[0, 0]);
}

fn scalar_dot(output: DotOutput<i32>) -> i32 {
    match output {
        DotOutput::Scalar(value) => value,
        DotOutput::Array(_) => panic!("expected scalar dot output"),
    }
}
