use crate::{ArrayElement, AtlasNdError, AtlasNdResult, view::ArrayView};

pub(crate) fn require_first_array<'arrays, 'data, T: ArrayElement>(
    arrays: &'arrays [ArrayView<'data, T>],
    op: &'static str,
) -> AtlasNdResult<&'arrays ArrayView<'data, T>> {
    arrays
        .first()
        .ok_or(AtlasNdError::InvalidArgument { op, reason: "at least one array is required" })
}

pub(crate) fn validate_matching_ndim<T: ArrayElement>(
    arrays: &[ArrayView<'_, T>],
    ndim: usize,
) -> AtlasNdResult<()> {
    for array in &arrays[1..] {
        if array.ndim() != ndim {
            return Err(AtlasNdError::DimensionMismatch { expected: ndim, actual: array.ndim() });
        }
    }

    Ok(())
}
