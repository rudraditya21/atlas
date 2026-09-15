use atlas_ndarray::{AtlasNdError, NDArray, SliceRange};

macro_rules! assert_numeric_reductions_match_scalar_reference {
    ($view:expr) => {{
        let view = $view;
        let values: Vec<f64> = view.iter().copied().collect();
        let expected_sum: f64 = values.iter().sum();
        let expected_product: f64 = values.iter().product();
        let expected_min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let expected_max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let expected_mean = expected_sum / values.len() as f64;

        assert_eq!(view.sum().unwrap(), expected_sum);
        assert_eq!(view.prod().unwrap(), expected_product);
        assert_eq!(view.min().unwrap(), expected_min);
        assert_eq!(view.max().unwrap(), expected_max);
        assert_eq!(view.mean().unwrap(), expected_mean);
    }};
}

macro_rules! assert_boolean_reductions_match_scalar_reference {
    ($view:expr) => {{
        let view = $view;
        let values: Vec<bool> = view.iter().copied().collect();

        assert_eq!(view.count_true(), values.iter().filter(|&&value| value).count());
        assert_eq!(view.all(), values.iter().all(|&value| value));
        assert_eq!(view.any(), values.iter().any(|&value| value));
    }};
}

#[test]
fn numeric_scalar_reductions_match_logical_values_across_layouts() {
    let contiguous =
        NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let padded_source =
        NDArray::from_shape_vec([2, 4], vec![1.0_f64, 2.0, 3.0, 99.0, 4.0, 5.0, 6.0, 99.0])
            .unwrap();
    let stepped_source =
        NDArray::from_shape_vec([2, 6], (1..=12).map(f64::from).collect()).unwrap();
    let scalar = NDArray::from_shape_vec([], vec![7.0_f64]).unwrap();

    assert_numeric_reductions_match_scalar_reference!(contiguous.view());
    assert_numeric_reductions_match_scalar_reference!(contiguous.view().transpose());
    assert_numeric_reductions_match_scalar_reference!(
        padded_source.view().slice([0, 0], [2, 3]).unwrap()
    );
    assert_numeric_reductions_match_scalar_reference!(
        stepped_source
            .view()
            .slice_ranges([SliceRange::full(), SliceRange::new(Some(0), Some(6), 2)])
            .unwrap()
    );
    assert_numeric_reductions_match_scalar_reference!(scalar.view());
}

#[test]
fn boolean_scalar_reductions_match_logical_values_across_layouts() {
    let contiguous =
        NDArray::from_shape_vec([2, 3], vec![true, false, true, true, true, false]).unwrap();
    let padded_source =
        NDArray::from_shape_vec([2, 4], vec![true, false, true, false, true, true, false, false])
            .unwrap();
    let stepped_source = NDArray::from_shape_vec(
        [2, 6],
        vec![true, false, true, false, true, false, false, true, true, false, true, true],
    )
    .unwrap();
    let scalar = NDArray::from_shape_vec([], vec![true]).unwrap();

    assert_boolean_reductions_match_scalar_reference!(contiguous.view());
    assert_boolean_reductions_match_scalar_reference!(contiguous.view().transpose());
    assert_boolean_reductions_match_scalar_reference!(
        padded_source.view().slice([0, 0], [2, 3]).unwrap()
    );
    assert_boolean_reductions_match_scalar_reference!(
        stepped_source
            .view()
            .slice_ranges([SliceRange::full(), SliceRange::new(Some(0), Some(6), 2)])
            .unwrap()
    );
    assert_boolean_reductions_match_scalar_reference!(scalar.view());
}

#[test]
fn scalar_reductions_preserve_empty_input_contracts() {
    let numeric = NDArray::<f64>::zeros([2, 0, 3]).unwrap();
    let boolean = NDArray::<bool>::full([2, 0, 3], false).unwrap();

    assert_eq!(numeric.sum().unwrap_err(), AtlasNdError::EmptyReduction { op: "sum" });
    assert_eq!(numeric.prod().unwrap_err(), AtlasNdError::EmptyReduction { op: "prod" });
    assert_eq!(numeric.min().unwrap_err(), AtlasNdError::EmptyReduction { op: "min" });
    assert_eq!(numeric.max().unwrap_err(), AtlasNdError::EmptyReduction { op: "max" });
    assert_eq!(numeric.mean().unwrap_err(), AtlasNdError::EmptyReduction { op: "mean" });
    assert_eq!(boolean.count_true(), 0);
    assert!(boolean.all());
    assert!(!boolean.any());
}

#[test]
fn scalar_reductions_preserve_nan_propagation_for_strided_views() {
    let values =
        NDArray::from_shape_vec([2, 3], vec![1.0_f64, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();
    let view = values.view().transpose();

    assert!(view.sum().unwrap().is_nan());
    assert!(view.prod().unwrap().is_nan());
    assert!(view.min().unwrap().is_nan());
    assert!(view.max().unwrap().is_nan());
    assert!(view.mean().unwrap().is_nan());
}
