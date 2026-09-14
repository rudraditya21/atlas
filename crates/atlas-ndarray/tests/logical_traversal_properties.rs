use atlas_ndarray::{NDArray, SliceRange};

#[test]
fn generated_strided_views_match_contiguous_logical_references() {
    let mut state = 0xC0FF_EE12_3456_7890_u64;

    for case_index in 0..64 {
        let dimensions = [
            (next(&mut state) % 4 + 3) as usize,
            (next(&mut state) % 4 + 3) as usize,
            (next(&mut state) % 4 + 3) as usize,
        ];
        let axis_cases = dimensions.map(|dimension| axis_case(dimension, &mut state));
        let values = (0..dimensions.iter().product())
            .map(|index| case_index as i32 * 1_000 + index as i32)
            .collect::<Vec<_>>();
        let array = NDArray::from_shape_vec(dimensions, values.clone()).unwrap();
        let ranges = axis_cases.map(|(start, step, length)| {
            SliceRange::new(
                Some(start as i64),
                Some((start + (length - 1) * step + 1) as i64),
                step as i64,
            )
        });
        let view = array.view().slice_ranges(ranges).unwrap().transpose();
        let expected = expected_transposed_slice(&values, dimensions, axis_cases);
        let reference = NDArray::from_shape_vec(view.shape().to_vec(), expected).unwrap();

        assert!(!view.is_contiguous(), "case {case_index} did not exercise strided traversal");
        assert_eq!(
            view.iter().copied().collect::<Vec<_>>(),
            reference.view().iter().copied().collect::<Vec<_>>()
        );
        assert_eq!(view.to_owned().data(), reference.data());
    }
}

fn axis_case(dimension: usize, state: &mut u64) -> (usize, usize, usize) {
    let step = next(state) as usize % 2 + 1;
    let start = next(state) as usize % (dimension - step);
    let maximum_length = (dimension - start - 1) / step + 1;
    let length = next(state) as usize % (maximum_length - 1) + 2;

    (start, step, length)
}

fn expected_transposed_slice(
    values: &[i32],
    dimensions: [usize; 3],
    axis_cases: [(usize, usize, usize); 3],
) -> Vec<i32> {
    let [(start0, step0, length0), (start1, step1, length1), (start2, step2, length2)] = axis_cases;
    let mut expected = Vec::with_capacity(length0 * length1 * length2);

    for index2 in 0..length2 {
        for index1 in 0..length1 {
            for index0 in 0..length0 {
                let row = start0 + index0 * step0;
                let column = start1 + index1 * step1;
                let depth = start2 + index2 * step2;
                expected.push(values[(row * dimensions[1] + column) * dimensions[2] + depth]);
            }
        }
    }

    expected
}

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    *state
}
