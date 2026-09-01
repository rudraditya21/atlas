use crate::{
    ArrayElement, AtlasNdError, AtlasNdResult, CastMode, NDArray, RuntimeScalar,
    internal::materialize_contiguous_array, view::ArrayView,
};

#[derive(Clone, Debug)]
pub enum AsArray<'a, T: ArrayElement> {
    Borrowed(ArrayView<'a, T>),
    Owned(NDArray<T>),
}

impl<'a, T: ArrayElement> AsArray<'a, T> {
    pub fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }

    pub fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
    }

    pub fn view(&self) -> ArrayView<'_, T> {
        match self {
            Self::Borrowed(view) => view.clone(),
            Self::Owned(array) => array.view(),
        }
    }

    pub fn into_owned(self) -> NDArray<T> {
        match self {
            Self::Borrowed(view) => materialize_contiguous_array(&view),
            Self::Owned(array) => array,
        }
    }
}

pub(crate) fn cast_array<T, U>(
    shape: Vec<usize>,
    values: impl IntoIterator<Item = T>,
    mode: CastMode,
) -> AtlasNdResult<NDArray<U>>
where
    T: RuntimeScalar,
    U: ArrayElement + RuntimeScalar,
{
    let from = T::dtype();
    let to = U::dtype();
    let values = values.into_iter();
    let mut casted = Vec::with_capacity(values.size_hint().0);

    for value in values {
        let value = value
            .into_scalar_value()
            .cast(to, mode)
            .and_then(U::from_scalar_value)
            .ok_or(AtlasNdError::InvalidCast { from, to, mode })?;
        casted.push(value);
    }

    NDArray::from_row_major_parts(shape, casted)
}

#[cfg(test)]
mod tests {
    use crate::{AsArray, NDArray};

    #[test]
    fn asarray_wraps_borrowed_views_and_owned_arrays_consistently() {
        let array = NDArray::from_shape_vec([2, 2], vec![1_i32, 2, 3, 4]).unwrap();
        let borrowed = AsArray::Borrowed(array.view());
        let owned = AsArray::Owned(array.clone());

        assert!(borrowed.is_borrowed());
        assert!(!borrowed.is_owned());
        assert_eq!(borrowed.view().shape(), &[2, 2]);
        assert_eq!(borrowed.view().data(), &[1, 2, 3, 4]);

        assert!(owned.is_owned());
        assert!(!owned.is_borrowed());
        assert_eq!(owned.view().shape(), &[2, 2]);
        assert_eq!(owned.view().data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn asarray_into_owned_materializes_borrowed_views_in_logical_order() {
        let array = NDArray::from_shape_vec([2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let borrowed = AsArray::Borrowed(array.view().transpose());
        let owned = borrowed.into_owned();

        assert_eq!(owned.shape(), &[3, 2]);
        assert_eq!(owned.strides(), &[2, 1]);
        assert_eq!(owned.data(), &[0, 3, 1, 4, 2, 5]);
    }
}
