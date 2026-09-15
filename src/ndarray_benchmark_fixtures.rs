use atlas_ndarray::NDArray;

pub const MASKED_FILL_SIZES: [usize; 4] = [1 << 10, 1 << 14, 1 << 18, 1 << 20];

pub fn masked_fill_inputs(size: usize) -> (NDArray<f64>, NDArray<bool>) {
    (
        NDArray::from_vec(vec![size], vec![1.0; size]).expect("benchmark fixture shape is valid"),
        NDArray::from_vec(vec![size], (0..size).map(|index| index % 3 == 0).collect())
            .expect("benchmark fixture shape is valid"),
    )
}
