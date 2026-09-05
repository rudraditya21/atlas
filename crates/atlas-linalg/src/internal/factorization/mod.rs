mod copy;
mod pivot;
mod tolerance;
mod validate;

pub(crate) use self::{
    copy::{copy_matrix_row_major, identity_matrix_data, zero_matrix_data},
    pivot::{find_pivot_row, swap_l_prefix_rows, swap_rows},
    tolerance::{dot_slice, is_symmetric, tolerance, vector_norm},
    validate::{validate_finite, validate_rank_two},
};
