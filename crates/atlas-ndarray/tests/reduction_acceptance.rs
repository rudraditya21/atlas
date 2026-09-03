use atlas_ndarray::{AtlasNdError, NDArray};

#[test]
fn reductions_match_reference_values_across_axes_and_keepdims() {
    let array = NDArray::from_shape_vec([2, 3, 2], (0_i32..12).collect()).unwrap();

    for axis in 0..3 {
        let negative = axis - 3;
        assert_eq!(array.sum_axis(axis).unwrap().data(), array.sum_axis(negative).unwrap().data());
        assert_eq!(
            array.prod_axis(axis).unwrap().data(),
            array.prod_axis(negative).unwrap().data()
        );
        assert_eq!(
            array.mean_axis(axis).unwrap().data(),
            array.mean_axis(negative).unwrap().data()
        );
        assert_eq!(
            array.sum_axis_keepdims(axis).unwrap().len(),
            array.sum_axis(axis).unwrap().len()
        );
    }
}

#[test]
fn reductions_match_logical_values_for_transposed_and_sliced_views() {
    let array = NDArray::from_shape_vec([3, 4], (0_i32..12).collect()).unwrap();
    let view = array.view().transpose().slice([1, 1], [3, 2]).unwrap();

    assert_eq!(view.sum().unwrap(), 48);
    assert_eq!(view.min().unwrap(), 5);
    assert_eq!(view.max().unwrap(), 11);
    assert_eq!(view.sum_axis(-1).unwrap().data(), &[14, 16, 18]);
    assert_eq!(view.argmin_axis(-1).unwrap().data(), &[0, 0, 0]);
}

#[test]
fn float_nan_and_stability_contracts_are_layout_independent() {
    let values =
        NDArray::from_shape_vec([2, 3], vec![1.0e16_f64, 1.0, -1.0e16, 2.0, f64::NAN, 3.0])
            .unwrap();

    assert_eq!(values.sum_axis(0).unwrap().data()[0], 1.0e16 + 2.0);
    assert!(values.min_axis(1).unwrap().data()[1].is_nan());
    assert!(values.max_axis(1).unwrap().data()[1].is_nan());
    assert!(values.mean_axis(1).unwrap().data()[1].is_nan());
    assert_eq!(values.argmin_axis(1).unwrap().data(), &[2, 1]);
}

#[test]
fn nan_extrema_and_arg_reductions_match_for_contiguous_and_strided_layouts() {
    let contiguous =
        NDArray::from_shape_vec([3, 2], vec![1.0_f64, 4.0, f64::NAN, 5.0, 3.0, f64::NAN]).unwrap();
    let source =
        NDArray::from_shape_vec([2, 3], vec![1.0_f64, f64::NAN, 3.0, 4.0, 5.0, f64::NAN]).unwrap();
    let strided = source.view().transpose();

    assert!(contiguous.min().unwrap().is_nan());
    assert!(contiguous.max().unwrap().is_nan());
    assert!(strided.min().unwrap().is_nan());
    assert!(strided.max().unwrap().is_nan());
    assert_eq!(contiguous.argmin().unwrap(), 2);
    assert_eq!(contiguous.argmax().unwrap(), 2);
    assert_eq!(strided.argmin().unwrap(), 2);
    assert_eq!(strided.argmax().unwrap(), 2);

    assert_eq!(contiguous.argmin_axis(1).unwrap().data(), &[0, 0, 1]);
    assert_eq!(contiguous.argmax_axis(1).unwrap().data(), &[1, 0, 1]);
    assert_eq!(strided.argmin_axis(1).unwrap().data(), &[0, 0, 1]);
    assert_eq!(strided.argmax_axis(1).unwrap().data(), &[1, 0, 1]);
    assert!(strided.min_axis(1).unwrap().data()[1].is_nan());
    assert!(strided.max_axis(1).unwrap().data()[2].is_nan());
}

#[test]
fn zero_sized_reduction_outputs_and_empty_lanes_are_distinct() {
    let output_empty = NDArray::<i32>::new([2, 0, 3], 1).unwrap();
    let empty_lane = NDArray::<i32>::new([0, 3], 1).unwrap();

    assert_eq!(output_empty.sum_axis(0).unwrap().shape(), &[0, 3]);
    assert!(output_empty.variance_axis(-1).unwrap().data().is_empty());
    assert_eq!(
        empty_lane.variance_axis(0).unwrap_err(),
        AtlasNdError::EmptyReduction { op: "variance" }
    );
    assert_eq!(
        empty_lane.stddev_axis(0).unwrap_err(),
        AtlasNdError::EmptyReduction { op: "stddev" }
    );
}
