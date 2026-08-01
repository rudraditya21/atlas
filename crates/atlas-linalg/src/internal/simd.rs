use atlas_ndarray::Numeric;

const CONTIGUOUS_LANES: usize = 8;

pub(crate) fn dot_contiguous<T: Numeric>(lhs: &[T], rhs: &[T]) -> T {
    debug_assert_eq!(lhs.len(), rhs.len());

    let len = lhs.len();
    let body_len = body_len(len);
    let mut acc0 = T::zero();
    let mut acc1 = T::zero();
    let mut acc2 = T::zero();
    let mut acc3 = T::zero();
    let mut acc4 = T::zero();
    let mut acc5 = T::zero();
    let mut acc6 = T::zero();
    let mut acc7 = T::zero();
    let mut index = 0;

    while index < body_len {
        acc0 += lhs[index] * rhs[index];
        acc1 += lhs[index + 1] * rhs[index + 1];
        acc2 += lhs[index + 2] * rhs[index + 2];
        acc3 += lhs[index + 3] * rhs[index + 3];
        acc4 += lhs[index + 4] * rhs[index + 4];
        acc5 += lhs[index + 5] * rhs[index + 5];
        acc6 += lhs[index + 6] * rhs[index + 6];
        acc7 += lhs[index + 7] * rhs[index + 7];
        index += CONTIGUOUS_LANES;
    }

    let mut total = acc0 + acc1;
    total += acc2 + acc3;
    total += acc4 + acc5;
    total += acc6 + acc7;

    while index < len {
        total += lhs[index] * rhs[index];
        index += 1;
    }

    total
}

pub(crate) fn scaled_accumulate_contiguous<T: Numeric>(output: &mut [T], input: &[T], scale: T) {
    debug_assert_eq!(output.len(), input.len());

    let len = output.len();
    let body_len = body_len(len);
    let mut index = 0;

    while index < body_len {
        output[index] += scale * input[index];
        output[index + 1] += scale * input[index + 1];
        output[index + 2] += scale * input[index + 2];
        output[index + 3] += scale * input[index + 3];
        output[index + 4] += scale * input[index + 4];
        output[index + 5] += scale * input[index + 5];
        output[index + 6] += scale * input[index + 6];
        output[index + 7] += scale * input[index + 7];
        index += CONTIGUOUS_LANES;
    }

    while index < len {
        output[index] += scale * input[index];
        index += 1;
    }
}

#[inline]
fn body_len(len: usize) -> usize {
    len / CONTIGUOUS_LANES * CONTIGUOUS_LANES
}
