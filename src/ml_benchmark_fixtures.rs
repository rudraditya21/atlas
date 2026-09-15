use atlas_ndarray::NDArray;

pub const SAMPLE_COUNTS: [usize; 3] = [64, 256, 1_024];
pub const FEATURE_COUNT: usize = 8;

pub fn features(sample_count: usize) -> NDArray<f64> {
    NDArray::from_shape_vec(
        [sample_count, FEATURE_COUNT],
        (0..sample_count * FEATURE_COUNT)
            .map(|index| {
                let sample = index / FEATURE_COUNT;
                let feature = index % FEATURE_COUNT;
                let class_offset = if sample < sample_count / 2 { -1.0 } else { 1.0 };
                if feature == 0 {
                    class_offset
                } else {
                    ((sample * (feature * 7 + 3)) % 31) as f64 / 31.0 - 0.5
                }
            })
            .collect(),
    )
    .expect("benchmark fixture shape is valid")
}

pub fn classification_labels(sample_count: usize) -> NDArray<usize> {
    NDArray::from_shape_vec(
        [sample_count],
        (0..sample_count).map(|sample| usize::from(sample >= sample_count / 2)).collect(),
    )
    .expect("benchmark fixture shape is valid")
}

pub fn regression_targets(features: &NDArray<f64>) -> NDArray<f64> {
    NDArray::from_shape_vec(
        [features.shape()[0]],
        (0..features.shape()[0])
            .map(|sample| {
                0.5 + (0..FEATURE_COUNT)
                    .map(|feature| {
                        features.data()[sample * FEATURE_COUNT + feature] * (feature + 1) as f64
                    })
                    .sum::<f64>()
            })
            .collect(),
    )
    .expect("benchmark fixture shape is valid")
}
