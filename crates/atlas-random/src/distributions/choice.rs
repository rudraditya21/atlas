use atlas_ndarray::{ArrayElement, NDArray, OperandMetadata, checked_element_count};

use crate::{
    core::{AtlasRandomError, AtlasRandomResult},
    rng::random_source::RandomSource,
};

/// Samples `sample_count` distinct indices from `0..population_size` in random order.
pub fn choice_indices<R: RandomSource>(
    population_size: usize,
    sample_count: usize,
    rng: &mut R,
) -> AtlasRandomResult<NDArray<usize>> {
    sample_indices(population_size, sample_count, rng, "choice_indices")
        .and_then(|indices| NDArray::from_shape_vec([sample_count], indices).map_err(Into::into))
}

/// Samples `sample_count` distinct logical positions from `input` without replacement.
///
/// Duplicate source values can appear when they occupy different logical positions.
pub fn choice<T, O, R>(input: &O, sample_count: usize, rng: &mut R) -> AtlasRandomResult<NDArray<T>>
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
    R: RandomSource,
{
    let population_size = checked_element_count(input.shape())?;
    let indices = sample_indices(population_size, sample_count, rng, "choice")?;
    let values =
        indices.into_iter().map(|index| input.data()[logical_offset(input, index)]).collect();

    NDArray::from_shape_vec([sample_count], values).map_err(Into::into)
}

fn sample_indices<R: RandomSource>(
    population_size: usize,
    sample_count: usize,
    rng: &mut R,
    op: &'static str,
) -> AtlasRandomResult<Vec<usize>> {
    if sample_count > population_size {
        return Err(AtlasRandomError::InvalidArgument {
            op,
            reason: "sample count must not exceed population size",
        });
    }

    let mut indices = (0..population_size).collect::<Vec<_>>();
    rng.shuffle(&mut indices);
    indices.truncate(sample_count);
    Ok(indices)
}

fn logical_offset<T, O>(input: &O, mut linear_index: usize) -> usize
where
    T: ArrayElement,
    O: OperandMetadata<T> + ?Sized,
{
    let mut offset = input.offset();
    for axis in (0..input.ndim()).rev() {
        let coordinate = linear_index % input.shape()[axis];
        linear_index /= input.shape()[axis];
        offset += coordinate * input.strides()[axis];
    }
    offset
}

#[cfg(test)]
mod tests {
    use atlas_ndarray::NDArray;

    use crate::{AtlasRandomError, AtlasRng, choice, choice_indices};

    #[test]
    fn choice_indices_are_seeded_and_unique() {
        let mut left_rng = AtlasRng::seed_from_u64(17);
        let mut right_rng = AtlasRng::seed_from_u64(17);
        let left = choice_indices(8, 3, &mut left_rng).unwrap();
        let right = choice_indices(8, 3, &mut right_rng).unwrap();
        let mut sorted = left.data().to_vec();
        sorted.sort_unstable();

        assert_eq!(left.data(), right.data());
        assert!(sorted.windows(2).all(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn choice_supports_full_populations() {
        let values = NDArray::from_shape_vec([4], vec![10_i32, 20, 30, 40]).unwrap();
        let mut rng = AtlasRng::seed_from_u64(23);
        let mut selected = choice(&values, 4, &mut rng).unwrap().data().to_vec();
        selected.sort_unstable();

        assert_eq!(selected, values.data());
    }

    #[test]
    fn choice_rejects_sample_counts_larger_than_the_population() {
        let values = NDArray::from_shape_vec([2], vec![1_i32, 2]).unwrap();
        let mut rng = AtlasRng::seed_from_u64(29);

        assert_eq!(
            choice(&values, 3, &mut rng).unwrap_err(),
            AtlasRandomError::InvalidArgument {
                op: "choice",
                reason: "sample count must not exceed population size",
            }
        );
    }

    #[test]
    fn choice_handles_empty_populations_when_no_values_are_requested() {
        let values = NDArray::<i32>::zeros([0]).unwrap();
        let mut rng = AtlasRng::seed_from_u64(31);

        assert!(choice_indices(0, 0, &mut rng).unwrap().data().is_empty());
        assert!(choice(&values, 0, &mut rng).unwrap().data().is_empty());
    }
}
