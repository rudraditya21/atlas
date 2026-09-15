use atlas_benchmarks::ndarray_benchmark_fixtures::{MASKED_FILL_SIZES, masked_fill_inputs};

#[test]
fn masked_fill_benchmark_fixtures_produce_expected_values() {
    for size in MASKED_FILL_SIZES {
        let (mut values, mask) = masked_fill_inputs(size);

        values.masked_fill(&mask, 0.0).unwrap();

        assert!(
            values
                .data()
                .iter()
                .enumerate()
                .all(|(index, value)| *value == if index % 3 == 0 { 0.0 } else { 1.0 })
        );
    }
}
