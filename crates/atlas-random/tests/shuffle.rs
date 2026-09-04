use atlas_ndarray::NDArray;
use atlas_random::{AtlasRng, shuffle_axis};

#[test]
fn shuffle_axis_is_seeded_shape_preserving_and_lossless() {
    let values = NDArray::from_shape_vec([3, 2], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let mut left = AtlasRng::seed_from_u64(47);
    let mut right = AtlasRng::seed_from_u64(47);
    let shuffled = shuffle_axis(&values, 0, &mut left).unwrap();
    let repeated = shuffle_axis(&values, 0, &mut right).unwrap();
    let mut sorted = shuffled.data().to_vec();
    sorted.sort_unstable();

    assert_eq!(shuffled.shape(), values.shape());
    assert_eq!(shuffled.data(), repeated.data());
    assert_eq!(sorted, values.data());
}

#[test]
fn shuffle_axis_supports_negative_axes_and_views() {
    let values = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
    let mut rng = AtlasRng::seed_from_u64(53);
    let shuffled = shuffle_axis(&values.view().transpose(), -1, &mut rng).unwrap();
    let mut sorted = shuffled.data().to_vec();
    sorted.sort_unstable();

    assert_eq!(shuffled.shape(), &[3, 2]);
    assert_eq!(sorted, values.data());
}
