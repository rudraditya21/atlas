use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn nan_reductions_ignore_nan_values_for_arrays_and_views() {
    let values = NDArray::from_shape_vec([2, 2], vec![f64::NAN, 1.0, 3.0, f64::NAN]).unwrap();
    let view = values.view().transpose();

    assert_eq!(values.nanmin().unwrap(), 1.0);
    assert_eq!(values.nanmax().unwrap(), 3.0);
    assert_eq!(values.nanmean().unwrap(), 2.0);
    assert_eq!(values.nanstd().unwrap(), 1.0);
    assert_eq!(view.nanmin().unwrap(), 1.0);
    assert_eq!(view.nanmax().unwrap(), 3.0);
    assert_eq!(view.nanmean().unwrap(), 2.0);
    assert_eq!(view.nanstd().unwrap(), 1.0);
}

#[test]
fn nan_reductions_distinguish_empty_and_all_nan_inputs() {
    let empty = NDArray::<f64>::zeros([0]).unwrap();
    let all_nan = NDArray::from_shape_vec([2], vec![f64::NAN, f64::NAN]).unwrap();

    assert_eq!(empty.nanmin().unwrap_err(), AtlasNdError::EmptyReduction { op: "nanmin" });
    assert_eq!(empty.nanmax().unwrap_err(), AtlasNdError::EmptyReduction { op: "nanmax" });
    assert_eq!(empty.nanmean().unwrap_err(), AtlasNdError::EmptyReduction { op: "nanmean" });
    assert_eq!(empty.nanstd().unwrap_err(), AtlasNdError::EmptyReduction { op: "nanstd" });
    assert_eq!(all_nan.nanmin().unwrap_err(), AtlasNdError::AllNaN { op: "nanmin" });
    assert_eq!(all_nan.nanmax().unwrap_err(), AtlasNdError::AllNaN { op: "nanmax" });
    assert_eq!(all_nan.nanmean().unwrap_err(), AtlasNdError::AllNaN { op: "nanmean" });
    assert_eq!(all_nan.nanstd().unwrap_err(), AtlasNdError::AllNaN { op: "nanstd" });
}
