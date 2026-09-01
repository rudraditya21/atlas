use atlas_ndarray::{ArrayView, NDArray};
use proptest::prelude::*;

const MAX_RANK: usize = 4;
const MAX_DIMENSION: usize = 4;

#[derive(Debug, Clone)]
pub(crate) struct ViewFixture {
    array: NDArray<i32>,
    transpose: bool,
}

impl ViewFixture {
    pub(crate) fn view(&self) -> ArrayView<'_, i32> {
        let view = self.array.view();
        let starts: Vec<_> = view.shape().iter().map(|dimension| dimension / 3).collect();
        let shape: Vec<_> =
            view.shape().iter().zip(&starts).map(|(dimension, start)| dimension - start).collect();
        let view = view.slice(&starts, shape).expect("fixture slices use in-bounds ranges");

        if self.transpose { view.transpose() } else { view }
    }
}

pub(crate) fn shape_strategy() -> impl Strategy<Value = Vec<usize>> {
    prop::collection::vec(0..=MAX_DIMENSION, 0..=MAX_RANK)
}

pub(crate) fn array_strategy() -> impl Strategy<Value = NDArray<i32>> {
    shape_strategy().prop_flat_map(|shape| {
        let len: usize = shape.iter().product();
        prop::collection::vec(any::<i32>(), len).prop_map(move |data| {
            NDArray::from_shape_vec(shape.clone(), data)
                .expect("generated data length matches its shape")
        })
    })
}

pub(crate) fn view_fixture_strategy() -> impl Strategy<Value = ViewFixture> {
    (array_strategy(), any::<bool>())
        .prop_map(|(array, transpose)| ViewFixture { array, transpose })
}

pub(crate) fn assert_owned_invariants<T>(array: &NDArray<T>)
where
    T: atlas_ndarray::ArrayElement,
{
    assert_eq!(array.data().len(), array.shape().iter().product());
    assert_eq!(array.shape().len(), array.strides().len());
    assert_eq!(array.len(), array.data().len());
}

pub(crate) fn assert_view_invariants<T>(view: &ArrayView<'_, T>)
where
    T: atlas_ndarray::ArrayElement + PartialEq,
{
    assert_eq!(view.shape().len(), view.strides().len());

    if view.is_empty() {
        assert!(view.offset() <= view.data().len());
    } else {
        let max_offset = view.offset()
            + view
                .shape()
                .iter()
                .zip(view.strides())
                .map(|(dimension, stride)| (dimension - 1) * stride)
                .sum::<usize>();
        assert!(max_offset < view.data().len());
    }

    let logical_values: Vec<_> = view.iter().copied().collect();
    let materialized = view.to_owned();
    assert_eq!(materialized.shape(), view.shape());
    assert_eq!(materialized.data(), logical_values);
}
