use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn scalar_and_zero_sized_constructors_preserve_layout_invariants() {
    let scalar = NDArray::full([], 3.5_f64).unwrap();
    let empty_1d = NDArray::<i32>::empty([0]).unwrap();
    let zero_dim = NDArray::<i32>::zeros([2, 0, 3]).unwrap();
    let identity = NDArray::<f64>::eye(0).unwrap();

    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(scalar.strides(), &[] as &[usize]);
    assert_eq!(scalar.data(), &[3.5]);

    assert_eq!(empty_1d.shape(), &[0]);
    assert_eq!(empty_1d.strides(), &[1]);
    assert!(empty_1d.data().is_empty());

    assert_eq!(zero_dim.shape(), &[2, 0, 3]);
    assert_eq!(zero_dim.len(), 0);
    assert!(zero_dim.data().is_empty());

    assert_eq!(identity.shape(), &[0, 0]);
    assert!(identity.data().is_empty());
}

#[test]
fn eye_is_ready_for_linalg_bootstrap() {
    let identity = NDArray::<f64>::eye(4).unwrap();

    assert_eq!(identity.shape(), &[4, 4]);
    assert_eq!(identity.get(&[0, 0]).unwrap(), &1.0);
    assert_eq!(identity.get(&[1, 2]).unwrap(), &0.0);
    assert_eq!(identity.get(&[3, 3]).unwrap(), &1.0);
}

#[test]
fn arange_creates_predictable_1d_workloads() {
    let values = NDArray::arange(0_i32, 6, 2).unwrap();
    let empty = NDArray::arange(4_i32, 4, 1).unwrap();
    let descending_with_positive_step = NDArray::arange(6_i32, 0, 2).unwrap();
    let ascending_with_negative_step = NDArray::arange(0_i32, 6, -2).unwrap();

    assert_eq!(values.shape(), &[3]);
    assert_eq!(values.data(), &[0, 2, 4]);
    assert_eq!(empty.shape(), &[0]);
    assert!(empty.data().is_empty());
    assert_eq!(descending_with_positive_step.shape(), &[0]);
    assert!(descending_with_positive_step.data().is_empty());
    assert_eq!(ascending_with_negative_step.shape(), &[0]);
    assert!(ascending_with_negative_step.data().is_empty());
}

#[test]
fn linspace_creates_evenly_spaced_floating_point_inputs() {
    let values = NDArray::linspace(-1.0_f32, 1.0, 5).unwrap();
    let descending = NDArray::linspace(2.0_f32, -2.0, 3).unwrap();
    let repeated = NDArray::linspace(1.25_f32, 1.25, 4).unwrap();
    let empty = NDArray::linspace(1.25_f32, 1.25, 0).unwrap();

    assert_eq!(values.shape(), &[5]);
    assert_eq!(values.data(), &[-1.0, -0.5, 0.0, 0.5, 1.0]);
    assert_eq!(descending.data(), &[2.0, 0.0, -2.0]);
    assert_eq!(repeated.data(), &[1.25, 1.25, 1.25, 1.25]);
    assert_eq!(empty.shape(), &[0]);
    assert_eq!(empty.strides(), &[1]);
    assert!(empty.data().is_empty());
}

#[test]
fn constructor_validation_is_consistent() {
    assert_eq!(
        NDArray::arange(0_i32, 5, 0).unwrap_err(),
        AtlasNdError::InvalidArgument { op: "arange", reason: "step must be non-zero" }
    );

    assert_eq!(
        NDArray::arange(f32::NEG_INFINITY, 1.0, 1.0).unwrap_err(),
        AtlasNdError::InvalidArgument { op: "arange", reason: "start end and step must be finite" }
    );

    assert_eq!(
        NDArray::<i32>::from_shape_vec([], Vec::new()).unwrap_err(),
        AtlasNdError::ShapeMismatch { expected: 1, actual: 0 }
    );

    assert_eq!(
        NDArray::linspace(f32::NEG_INFINITY, 1.0, 4).unwrap_err(),
        AtlasNdError::InvalidArgument { op: "linspace", reason: "start and end must be finite" }
    );
}
