use crate::{ArrayElement, ArrayView, AtlasNdError, AtlasNdResult, NDArray};

impl<T: ArrayElement> NDArray<T> {
    pub fn diagonal(&self, offset: isize) -> AtlasNdResult<ArrayView<'_, T>> {
        self.view().diagonal(offset)
    }
}

impl<'a, T: ArrayElement> ArrayView<'a, T> {
    pub fn diagonal(&self, offset: isize) -> AtlasNdResult<ArrayView<'a, T>> {
        if self.ndim() != 2 {
            return Err(AtlasNdError::InvalidArgument {
                op: "diagonal",
                reason: "array must be two-dimensional",
            });
        }

        let (row, column) =
            if offset >= 0 { (0, offset as usize) } else { (offset.unsigned_abs(), 0) };
        let length = self.shape[0].saturating_sub(row).min(self.shape[1].saturating_sub(column));
        let stride = if length == 0 {
            1
        } else {
            self.strides[0].checked_add(self.strides[1]).ok_or_else(|| {
                AtlasNdError::ShapeOverflow { op: "diagonal", shape: self.shape.clone() }
            })?
        };
        let offset = if length == 0 {
            self.offset
        } else {
            let row_offset = row.checked_mul(self.strides[0]).ok_or_else(|| {
                AtlasNdError::ShapeOverflow { op: "diagonal", shape: self.shape.clone() }
            })?;
            let column_offset = column.checked_mul(self.strides[1]).ok_or_else(|| {
                AtlasNdError::ShapeOverflow { op: "diagonal", shape: self.shape.clone() }
            })?;

            self.offset
                .checked_add(row_offset)
                .and_then(|offset| offset.checked_add(column_offset))
                .ok_or_else(|| AtlasNdError::ShapeOverflow {
                    op: "diagonal",
                    shape: self.shape.clone(),
                })?
        };

        ArrayView::from_parts(self.data, offset, vec![length], vec![stride])
    }
}
