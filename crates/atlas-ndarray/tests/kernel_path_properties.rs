use atlas_ndarray::NDArray;
use proptest::prelude::*;

fn matching_contiguous_and_strided() -> impl Strategy<Value = (NDArray<i32>, NDArray<i32>)> {
    (1_usize..=4, 1_usize..=4).prop_flat_map(|(rows, columns)| {
        prop::collection::vec(any::<i32>(), rows * columns).prop_map(move |logical| {
            let mut backing = vec![0; logical.len()];
            for row in 0..rows {
                for column in 0..columns {
                    backing[column * rows + row] = logical[row * columns + column];
                }
            }
            (
                NDArray::from_shape_vec([rows, columns], logical).unwrap(),
                NDArray::from_shape_vec([columns, rows], backing).unwrap(),
            )
        })
    })
}

proptest! {
    #[test]
    fn contiguous_and_strided_elementwise_kernels_agree(
        (contiguous, backing) in matching_contiguous_and_strided(),
    ) {
        let strided = backing.view().transpose();

        prop_assert_eq!(contiguous.abs().data(), strided.abs().data());
        prop_assert_eq!(contiguous.sign().data(), strided.sign().data());
    }

    #[test]
    fn contiguous_and_strided_reduction_kernels_agree(
        (contiguous, backing) in matching_contiguous_and_strided(),
    ) {
        let strided = backing.view().transpose();

        prop_assert_eq!(contiguous.sum().unwrap(), strided.sum().unwrap());
        prop_assert_eq!(contiguous.prod().unwrap(), strided.prod().unwrap());
        let contiguous_sum = contiguous.sum_axis(1_i32).unwrap();
        let strided_sum = strided.sum_axis(1_i32).unwrap();
        let contiguous_product = contiguous.prod_axis(0_i32).unwrap();
        let strided_product = strided.prod_axis(0_i32).unwrap();

        prop_assert_eq!(contiguous_sum.data(), strided_sum.data());
        prop_assert_eq!(contiguous_product.data(), strided_product.data());
    }
}
