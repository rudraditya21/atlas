use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn eye_is_ready_for_linalg_bootstrap() {
    let identity = NDArray::<f64>::eye(4);

    assert_eq!(identity.shape(), &[4, 4]);
    assert_eq!(identity.get(&[0, 0]).unwrap(), &1.0);
    assert_eq!(identity.get(&[1, 2]).unwrap(), &0.0);
    assert_eq!(identity.get(&[3, 3]).unwrap(), &1.0);
}

#[test]
fn arange_creates_predictable_1d_workloads() {
    let values = NDArray::arange(0_i32, 6, 2).unwrap();

    assert_eq!(values.shape(), &[3]);
    assert_eq!(values.data(), &[0, 2, 4]);
}

#[test]
fn linspace_creates_evenly_spaced_floating_point_inputs() {
    let values = NDArray::linspace(-1.0_f32, 1.0, 5).unwrap();

    assert_eq!(values.shape(), &[5]);
    assert_eq!(values.data(), &[-1.0, -0.5, 0.0, 0.5, 1.0]);
}

#[test]
fn constructor_validation_is_consistent() {
    assert_eq!(
        NDArray::arange(0_i32, 5, 0).unwrap_err(),
        AtlasNdError::InvalidArgument {
            op: "arange",
            reason: "step must be non-zero",
        }
    );
}
