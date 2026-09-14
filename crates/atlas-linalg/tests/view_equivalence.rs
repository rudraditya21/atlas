use atlas_linalg::{DotOutput, batched_diag, batched_transpose, dot, matmul, norm, trace};
use atlas_ndarray::NDArray;

#[test]
fn dense_operations_match_equivalent_logical_views() {
    let left = NDArray::from_shape_vec([2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    let left_view_source = NDArray::from_shape_vec([3, 2], vec![1_i32, 4, 2, 5, 3, 6]).unwrap();
    let right = NDArray::from_shape_vec([3, 2], vec![7_i32, 8, 9, 10, 11, 12]).unwrap();
    let right_view_source = NDArray::from_shape_vec([2, 3], vec![7_i32, 9, 11, 8, 10, 12]).unwrap();
    let vector = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
    let vector_view_source = NDArray::from_shape_vec([4], vec![1_i32, 2, 3, 0]).unwrap();
    let square = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
    let square_view_source = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 3.0, 2.0, 4.0]).unwrap();
    let batched = NDArray::from_shape_vec([1, 2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    let batched_view_source =
        NDArray::from_shape_vec([1, 2, 4], vec![1_i32, 2, 3, 0, 4, 5, 6, 0]).unwrap();

    assert_eq!(
        matmul(&left, &right).unwrap().data(),
        matmul(left_view_source.view().transpose(), right_view_source.view().transpose())
            .unwrap()
            .data()
    );
    assert_eq!(
        matmul(&left, &vector).unwrap().data(),
        matmul(
            left_view_source.view().transpose(),
            vector_view_source.view().slice([0], [3]).unwrap()
        )
        .unwrap()
        .data()
    );
    assert_eq!(
        scalar_dot(dot(&vector, &vector).unwrap()),
        scalar_dot(
            dot(
                vector_view_source.view().slice([0], [3]).unwrap(),
                vector_view_source.view().slice([0], [3]).unwrap(),
            )
            .unwrap()
        )
    );
    assert_eq!(trace(&square).unwrap(), trace(square_view_source.view().transpose()).unwrap());
    assert!(
        (norm(&square).unwrap() - norm(square_view_source.view().transpose()).unwrap()).abs()
            < 1e-12
    );
    assert_eq!(
        batched_transpose(&batched).unwrap().data(),
        batched_transpose(batched_view_source.view().slice([0, 0, 0], [1, 2, 3]).unwrap())
            .unwrap()
            .data()
    );
    assert_eq!(
        batched_diag(&batched, 0).unwrap().data(),
        batched_diag(batched_view_source.view().slice([0, 0, 0], [1, 2, 3]).unwrap(), 0)
            .unwrap()
            .data()
    );
}

#[test]
fn vector_dot_outputs_are_equivalent_for_contiguous_and_sliced_views() {
    let values = NDArray::from_shape_vec([3], vec![1_i32, 2, 3]).unwrap();
    let source = NDArray::from_shape_vec([4], vec![1_i32, 2, 3, 0]).unwrap();

    assert_eq!(scalar_dot(dot(&values, &values).unwrap()), 14);
    assert_eq!(scalar_dot(dot(source.view().slice([0], [3]).unwrap(), &values).unwrap()), 14);
}

fn scalar_dot(output: DotOutput<i32>) -> i32 {
    match output {
        DotOutput::Scalar(value) => value,
        DotOutput::Array(_) => panic!("expected scalar dot output"),
    }
}
