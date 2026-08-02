use atlas_ndarray::Numeric;
use rayon::prelude::*;

use crate::internal::dense::{MatrixRef, VectorRef, dot_contiguous};
use crate::internal::simd;

use super::dispatch::should_parallelize_matmul;

pub(super) fn vector_matrix<T: Numeric>(lhs: VectorRef<'_, T>, rhs: MatrixRef<'_, T>) -> Vec<T> {
    let lhs = lhs.contiguous_slice();
    let mut data = vec![T::zero(); rhs.cols];

    if simd::is_f32::<T>() {
        let lhs = simd::cast_slice::<T, f32>(lhs);
        {
            let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

            for (col, output) in data_f32.iter_mut().enumerate() {
                *output = simd::dot_contiguous_f32(
                    lhs,
                    simd::cast_slice::<T, f32>(rhs.contiguous_col_slice(col)),
                );
            }
        }

        return data;
    }

    if simd::is_f64::<T>() {
        let lhs = simd::cast_slice::<T, f64>(lhs);
        {
            let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

            for (col, output) in data_f64.iter_mut().enumerate() {
                *output = simd::dot_contiguous_f64(
                    lhs,
                    simd::cast_slice::<T, f64>(rhs.contiguous_col_slice(col)),
                );
            }
        }

        return data;
    }

    for (col, output) in data.iter_mut().enumerate() {
        *output = dot_contiguous(lhs, rhs.contiguous_col_slice(col));
    }

    data
}

pub(super) fn matrix_vector<T: Numeric>(lhs: MatrixRef<'_, T>, rhs: VectorRef<'_, T>) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows];

    if simd::is_f32::<T>() {
        let rhs_values = simd::cast_slice::<T, f32>(rhs.contiguous_slice());
        {
            let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

            for (k, &rhs_value) in rhs_values.iter().enumerate() {
                let lhs_col = simd::cast_slice::<T, f32>(lhs.contiguous_col_slice(k));
                simd::scaled_accumulate_contiguous_f32(data_f32, lhs_col, rhs_value);
            }
        }

        return data;
    }

    if simd::is_f64::<T>() {
        let rhs_values = simd::cast_slice::<T, f64>(rhs.contiguous_slice());
        {
            let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

            for (k, &rhs_value) in rhs_values.iter().enumerate() {
                let lhs_col = simd::cast_slice::<T, f64>(lhs.contiguous_col_slice(k));
                simd::scaled_accumulate_contiguous_f64(data_f64, lhs_col, rhs_value);
            }
        }

        return data;
    }

    for k in 0..lhs.cols {
        simd::scaled_accumulate_contiguous(&mut data, lhs.contiguous_col_slice(k), rhs.value_at(k));
    }

    data
}

pub(super) fn matrix_matrix_lhs<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows * rhs.cols];

    if simd::is_f32::<T>() {
        return matrix_matrix_lhs_f32(lhs, rhs, data);
    }

    if simd::is_f64::<T>() {
        return matrix_matrix_lhs_f64(lhs, rhs, data);
    }

    if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
        data.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
            for k in 0..lhs.cols {
                let lhs_value = lhs.contiguous_col_slice(k)[row];
                let rhs_row = rhs.contiguous_row_slice(k);
                simd::scaled_accumulate_contiguous(out_row, rhs_row, lhs_value);
            }
        });
    } else {
        for k in 0..lhs.cols {
            let lhs_col = lhs.contiguous_col_slice(k);
            let rhs_row = rhs.contiguous_row_slice(k);

            for row in 0..lhs.rows {
                let lhs_value = lhs_col[row];
                let out_row = &mut data[row * rhs.cols..(row + 1) * rhs.cols];
                simd::scaled_accumulate_contiguous(out_row, rhs_row, lhs_value);
            }
        }
    }

    data
}

pub(super) fn matrix_matrix_rhs<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
) -> Vec<T> {
    let mut data = vec![T::zero(); lhs.rows * rhs.cols];

    if simd::is_f32::<T>() {
        return matrix_matrix_rhs_f32(lhs, rhs, data);
    }

    if simd::is_f64::<T>() {
        return matrix_matrix_rhs_f64(lhs, rhs, data);
    }

    if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
        data.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
            let lhs_row = lhs.contiguous_row_slice(row);

            for (col, output) in out_row.iter_mut().enumerate() {
                *output = dot_contiguous(lhs_row, rhs.contiguous_col_slice(col));
            }
        });
    } else {
        for row in 0..lhs.rows {
            let lhs_row = lhs.contiguous_row_slice(row);
            let out_row = &mut data[row * rhs.cols..(row + 1) * rhs.cols];

            for (col, output) in out_row.iter_mut().enumerate() {
                *output = dot_contiguous(lhs_row, rhs.contiguous_col_slice(col));
            }
        }
    }

    data
}

fn matrix_matrix_lhs_f32<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    mut data: Vec<T>,
) -> Vec<T> {
    {
        let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

        if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
            data_f32.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
                for k in 0..lhs.cols {
                    let lhs_value = simd::cast_slice::<T, f32>(lhs.contiguous_col_slice(k))[row];
                    let rhs_row = simd::cast_slice::<T, f32>(rhs.contiguous_row_slice(k));
                    simd::scaled_accumulate_contiguous_f32(out_row, rhs_row, lhs_value);
                }
            });
        } else {
            for k in 0..lhs.cols {
                let lhs_col = simd::cast_slice::<T, f32>(lhs.contiguous_col_slice(k));
                let rhs_row = simd::cast_slice::<T, f32>(rhs.contiguous_row_slice(k));

                for row in 0..lhs.rows {
                    let out_row = &mut data_f32[row * rhs.cols..(row + 1) * rhs.cols];
                    simd::scaled_accumulate_contiguous_f32(out_row, rhs_row, lhs_col[row]);
                }
            }
        }
    }

    data
}

fn matrix_matrix_lhs_f64<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    mut data: Vec<T>,
) -> Vec<T> {
    {
        let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

        if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
            data_f64.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
                for k in 0..lhs.cols {
                    let lhs_value = simd::cast_slice::<T, f64>(lhs.contiguous_col_slice(k))[row];
                    let rhs_row = simd::cast_slice::<T, f64>(rhs.contiguous_row_slice(k));
                    simd::scaled_accumulate_contiguous_f64(out_row, rhs_row, lhs_value);
                }
            });
        } else {
            for k in 0..lhs.cols {
                let lhs_col = simd::cast_slice::<T, f64>(lhs.contiguous_col_slice(k));
                let rhs_row = simd::cast_slice::<T, f64>(rhs.contiguous_row_slice(k));

                for row in 0..lhs.rows {
                    let out_row = &mut data_f64[row * rhs.cols..(row + 1) * rhs.cols];
                    simd::scaled_accumulate_contiguous_f64(out_row, rhs_row, lhs_col[row]);
                }
            }
        }
    }

    data
}

fn matrix_matrix_rhs_f32<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    mut data: Vec<T>,
) -> Vec<T> {
    {
        let data_f32 = simd::cast_mut_slice::<T, f32>(data.as_mut_slice());

        if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
            data_f32.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
                let lhs_row = simd::cast_slice::<T, f32>(lhs.contiguous_row_slice(row));

                for (col, output) in out_row.iter_mut().enumerate() {
                    *output = simd::dot_contiguous_f32(
                        lhs_row,
                        simd::cast_slice::<T, f32>(rhs.contiguous_col_slice(col)),
                    );
                }
            });
        } else {
            for row in 0..lhs.rows {
                let lhs_row = simd::cast_slice::<T, f32>(lhs.contiguous_row_slice(row));
                let out_row = &mut data_f32[row * rhs.cols..(row + 1) * rhs.cols];

                for (col, output) in out_row.iter_mut().enumerate() {
                    *output = simd::dot_contiguous_f32(
                        lhs_row,
                        simd::cast_slice::<T, f32>(rhs.contiguous_col_slice(col)),
                    );
                }
            }
        }
    }

    data
}

fn matrix_matrix_rhs_f64<T: Numeric>(
    lhs: MatrixRef<'_, T>,
    rhs: MatrixRef<'_, T>,
    mut data: Vec<T>,
) -> Vec<T> {
    {
        let data_f64 = simd::cast_mut_slice::<T, f64>(data.as_mut_slice());

        if should_parallelize_matmul(lhs.rows, lhs.cols, rhs.cols) {
            data_f64.par_chunks_mut(rhs.cols).enumerate().for_each(|(row, out_row)| {
                let lhs_row = simd::cast_slice::<T, f64>(lhs.contiguous_row_slice(row));

                for (col, output) in out_row.iter_mut().enumerate() {
                    *output = simd::dot_contiguous_f64(
                        lhs_row,
                        simd::cast_slice::<T, f64>(rhs.contiguous_col_slice(col)),
                    );
                }
            });
        } else {
            for row in 0..lhs.rows {
                let lhs_row = simd::cast_slice::<T, f64>(lhs.contiguous_row_slice(row));
                let out_row = &mut data_f64[row * rhs.cols..(row + 1) * rhs.cols];

                for (col, output) in out_row.iter_mut().enumerate() {
                    *output = simd::dot_contiguous_f64(
                        lhs_row,
                        simd::cast_slice::<T, f64>(rhs.contiguous_col_slice(col)),
                    );
                }
            }
        }
    }

    data
}
