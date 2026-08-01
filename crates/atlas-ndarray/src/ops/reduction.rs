use num_traits::ToPrimitive;

use crate::{
    AtlasNdError, AtlasNdResult, AxisIndex, NDArray, Numeric,
    core::axis::normalize_axis,
    internal::{
        LayoutKind, for_each_value, is_contiguous_layout, offset_iter, simd, try_for_each_value,
    },
    layout::element_count,
    view::ArrayView,
};

impl<T: Numeric> NDArray<T> {
    pub fn sum(&self) -> T {
        if self.is_contiguous() {
            return sum_contiguous(&self.data);
        }

        sum_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn prod(&self) -> T {
        if self.is_contiguous() {
            return prod_contiguous(&self.data);
        }

        prod_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn min(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        if self.is_contiguous() {
            return min_contiguous(&self.data, "min");
        }

        min_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn max(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        if self.is_contiguous() {
            return max_contiguous(&self.data, "max");
        }

        max_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn mean(&self) -> AtlasNdResult<f64>
    where
        T: ToPrimitive,
    {
        if self.is_contiguous() {
            return mean_contiguous(&self.data, "mean");
        }

        mean_all(&self.data, 0, &self.shape, &self.strides)
    }

    pub fn sum_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self> {
        sum_axis_impl(&self.data, 0, &self.shape, &self.strides, axis)
    }

    pub fn prod_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self> {
        prod_axis_impl(&self.data, 0, &self.shape, &self.strides, axis)
    }

    pub fn min_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self>
    where
        T: PartialOrd,
    {
        min_axis_impl(&self.data, 0, &self.shape, &self.strides, axis)
    }

    pub fn max_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<Self>
    where
        T: PartialOrd,
    {
        max_axis_impl(&self.data, 0, &self.shape, &self.strides, axis)
    }

    pub fn mean_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<f64>>
    where
        T: ToPrimitive,
    {
        mean_axis_impl(&self.data, 0, &self.shape, &self.strides, axis)
    }
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    pub fn sum(&self) -> T {
        if self.is_contiguous() {
            return sum_contiguous(self.contiguous_slice());
        }

        sum_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn prod(&self) -> T {
        if self.is_contiguous() {
            return prod_contiguous(self.contiguous_slice());
        }

        prod_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn min(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        if self.is_contiguous() {
            return min_contiguous(self.contiguous_slice(), "min");
        }

        min_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn max(&self) -> AtlasNdResult<T>
    where
        T: PartialOrd,
    {
        if self.is_contiguous() {
            return max_contiguous(self.contiguous_slice(), "max");
        }

        max_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn mean(&self) -> AtlasNdResult<f64>
    where
        T: ToPrimitive,
    {
        if self.is_contiguous() {
            return mean_contiguous(self.contiguous_slice(), "mean");
        }

        mean_all(self.data, self.offset, &self.shape, &self.strides)
    }

    pub fn sum_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>> {
        sum_axis_impl(self.data, self.offset, &self.shape, &self.strides, axis)
    }

    pub fn prod_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>> {
        prod_axis_impl(self.data, self.offset, &self.shape, &self.strides, axis)
    }

    pub fn min_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>>
    where
        T: PartialOrd,
    {
        min_axis_impl(self.data, self.offset, &self.shape, &self.strides, axis)
    }

    pub fn max_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<T>>
    where
        T: PartialOrd,
    {
        max_axis_impl(self.data, self.offset, &self.shape, &self.strides, axis)
    }

    pub fn mean_axis<A: AxisIndex>(&self, axis: A) -> AtlasNdResult<NDArray<f64>>
    where
        T: ToPrimitive,
    {
        mean_axis_impl(self.data, self.offset, &self.shape, &self.strides, axis)
    }
}

impl<'a, T: Numeric> ArrayView<'a, T> {
    fn contiguous_slice(&self) -> &[T] {
        debug_assert!(self.is_contiguous());
        let len = element_count(&self.shape);
        &self.data[self.offset..self.offset + len]
    }
}

fn sum_contiguous<T: Numeric>(values: &[T]) -> T {
    simd::sum_contiguous(values)
}

fn prod_contiguous<T: Numeric>(values: &[T]) -> T {
    simd::prod_contiguous(values)
}

fn min_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    simd::min_contiguous(values, op)
}

fn max_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    simd::max_contiguous(values, op)
}

fn mean_contiguous<T>(values: &[T], op: &'static str) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
{
    simd::mean_contiguous(values, op)
}

fn sum_axis_impl<T: Numeric>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>> {
    let metadata = axis_reduction_metadata(shape, strides, axis)?;
    let source_layout = metadata.source_layout;
    let axis_layout = metadata.axis_layout;

    match (source_layout, axis_layout) {
        (LayoutKind::Contiguous, _) => sum_axis_dense_contiguous(data, base_offset, metadata),
        (LayoutKind::Strided, LayoutKind::Contiguous) => {
            sum_axis_contiguous(data, base_offset, metadata)
        }
        (LayoutKind::Strided, LayoutKind::Strided) => sum_axis_strided(data, base_offset, metadata),
    }
}

fn prod_axis_impl<T: Numeric>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>> {
    let metadata = axis_reduction_metadata(shape, strides, axis)?;
    let source_layout = metadata.source_layout;
    let axis_layout = metadata.axis_layout;

    match (source_layout, axis_layout) {
        (LayoutKind::Contiguous, _) => prod_axis_dense_contiguous(data, base_offset, metadata),
        (LayoutKind::Strided, LayoutKind::Contiguous) => {
            prod_axis_contiguous(data, base_offset, metadata)
        }
        (LayoutKind::Strided, LayoutKind::Strided) => {
            prod_axis_strided(data, base_offset, metadata)
        }
    }
}

fn min_axis_impl<T>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let metadata = axis_reduction_metadata(shape, strides, axis)?;
    if metadata.axis_len == 0 {
        return Err(AtlasNdError::EmptyReduction { op: "min" });
    }
    let source_layout = metadata.source_layout;
    let axis_layout = metadata.axis_layout;

    match (source_layout, axis_layout) {
        (LayoutKind::Contiguous, _) => min_axis_dense_contiguous(data, base_offset, metadata),
        (LayoutKind::Strided, LayoutKind::Contiguous) => {
            min_axis_contiguous(data, base_offset, metadata)
        }
        (LayoutKind::Strided, LayoutKind::Strided) => min_axis_strided(data, base_offset, metadata),
    }
}

fn max_axis_impl<T>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let metadata = axis_reduction_metadata(shape, strides, axis)?;
    if metadata.axis_len == 0 {
        return Err(AtlasNdError::EmptyReduction { op: "max" });
    }
    let source_layout = metadata.source_layout;
    let axis_layout = metadata.axis_layout;

    match (source_layout, axis_layout) {
        (LayoutKind::Contiguous, _) => max_axis_dense_contiguous(data, base_offset, metadata),
        (LayoutKind::Strided, LayoutKind::Contiguous) => {
            max_axis_contiguous(data, base_offset, metadata)
        }
        (LayoutKind::Strided, LayoutKind::Strided) => max_axis_strided(data, base_offset, metadata),
    }
}

fn mean_axis_impl<T>(
    data: &[T],
    base_offset: usize,
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
{
    let metadata = axis_reduction_metadata(shape, strides, axis)?;
    if metadata.axis_len == 0 {
        return Err(AtlasNdError::EmptyReduction { op: "mean" });
    }
    let source_layout = metadata.source_layout;
    let axis_layout = metadata.axis_layout;

    match (source_layout, axis_layout) {
        (LayoutKind::Contiguous, _) => mean_axis_dense_contiguous(data, base_offset, metadata),
        (LayoutKind::Strided, LayoutKind::Contiguous) => {
            mean_axis_contiguous(data, base_offset, metadata)
        }
        (LayoutKind::Strided, LayoutKind::Strided) => {
            mean_axis_strided(data, base_offset, metadata)
        }
    }
}

struct AxisReductionMetadata {
    output_shape: Vec<usize>,
    outer_strides: Vec<usize>,
    output_len: usize,
    axis_stride: usize,
    axis_len: usize,
    contiguous_outer_len: usize,
    contiguous_inner_len: usize,
    source_layout: LayoutKind,
    axis_layout: LayoutKind,
}

fn axis_reduction_metadata(
    shape: &[usize],
    strides: &[usize],
    axis: impl AxisIndex,
) -> AtlasNdResult<AxisReductionMetadata> {
    let axis = normalize_axis(axis, shape.len())?;

    let mut output_shape = Vec::with_capacity(shape.len().saturating_sub(1));
    let mut outer_strides = Vec::with_capacity(strides.len().saturating_sub(1));

    for (current_axis, (&dim, &stride)) in shape.iter().zip(strides.iter()).enumerate() {
        if current_axis == axis {
            continue;
        }

        output_shape.push(dim);
        outer_strides.push(stride);
    }

    let output_len = element_count(&output_shape);
    let contiguous_outer_len = element_count(&shape[..axis]);
    let contiguous_inner_len = element_count(&shape[axis + 1..]);

    Ok(AxisReductionMetadata {
        output_shape,
        outer_strides,
        output_len,
        axis_stride: strides[axis],
        axis_len: shape[axis],
        contiguous_outer_len,
        contiguous_inner_len,
        source_layout: if is_contiguous_layout(shape, strides) {
            LayoutKind::Contiguous
        } else {
            LayoutKind::Strided
        },
        axis_layout: if strides[axis] == 1 { LayoutKind::Contiguous } else { LayoutKind::Strided },
    })
}

fn sum_axis_dense_contiguous<T: Numeric>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let values =
        contiguous_region(data, base_offset, metadata.output_len.saturating_mul(metadata.axis_len));
    let mut reduced = vec![T::zero(); metadata.output_len];

    for outer in 0..metadata.contiguous_outer_len {
        let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;
        let output_start = outer * metadata.contiguous_inner_len;

        for inner in 0..metadata.contiguous_inner_len {
            let mut total = T::zero();
            let mut offset = block_start + inner;

            for _ in 0..metadata.axis_len {
                total += values[offset];
                offset += metadata.contiguous_inner_len;
            }

            reduced[output_start + inner] = total;
        }
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn prod_axis_dense_contiguous<T: Numeric>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let values =
        contiguous_region(data, base_offset, metadata.output_len.saturating_mul(metadata.axis_len));
    let mut reduced = vec![T::zero(); metadata.output_len];

    for outer in 0..metadata.contiguous_outer_len {
        let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;
        let output_start = outer * metadata.contiguous_inner_len;

        for inner in 0..metadata.contiguous_inner_len {
            let mut total = T::one();
            let mut offset = block_start + inner;

            for _ in 0..metadata.axis_len {
                total *= values[offset];
                offset += metadata.contiguous_inner_len;
            }

            reduced[output_start + inner] = total;
        }
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn min_axis_dense_contiguous<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let values =
        contiguous_region(data, base_offset, metadata.output_len.saturating_mul(metadata.axis_len));
    let mut reduced = vec![T::zero(); metadata.output_len];

    for outer in 0..metadata.contiguous_outer_len {
        let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;
        let output_start = outer * metadata.contiguous_inner_len;

        for inner in 0..metadata.contiguous_inner_len {
            let mut offset = block_start + inner;
            let mut current = values[offset];
            offset += metadata.contiguous_inner_len;

            for _ in 1..metadata.axis_len {
                let value = values[offset];
                if value < current {
                    current = value;
                }
                offset += metadata.contiguous_inner_len;
            }

            reduced[output_start + inner] = current;
        }
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn max_axis_dense_contiguous<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let values =
        contiguous_region(data, base_offset, metadata.output_len.saturating_mul(metadata.axis_len));
    let mut reduced = vec![T::zero(); metadata.output_len];

    for outer in 0..metadata.contiguous_outer_len {
        let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;
        let output_start = outer * metadata.contiguous_inner_len;

        for inner in 0..metadata.contiguous_inner_len {
            let mut offset = block_start + inner;
            let mut current = values[offset];
            offset += metadata.contiguous_inner_len;

            for _ in 1..metadata.axis_len {
                let value = values[offset];
                if value > current {
                    current = value;
                }
                offset += metadata.contiguous_inner_len;
            }

            reduced[output_start + inner] = current;
        }
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn mean_axis_dense_contiguous<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
{
    let values =
        contiguous_region(data, base_offset, metadata.output_len.saturating_mul(metadata.axis_len));
    let mut reduced = vec![0.0_f64; metadata.output_len];

    for outer in 0..metadata.contiguous_outer_len {
        let block_start = outer * metadata.axis_len * metadata.contiguous_inner_len;
        let output_start = outer * metadata.contiguous_inner_len;

        for inner in 0..metadata.contiguous_inner_len {
            let mut total = 0.0_f64;
            let mut offset = block_start + inner;

            for _ in 0..metadata.axis_len {
                total += values[offset]
                    .to_f64()
                    .ok_or(AtlasNdError::NumericConversionFailed { op: "mean" })?;
                offset += metadata.contiguous_inner_len;
            }

            reduced[output_start + inner] = total / metadata.axis_len as f64;
        }
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn sum_axis_contiguous<T: Numeric>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let mut reduced = vec![T::zero(); metadata.output_len];

    for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
        base_offset,
        &metadata.output_shape,
        &metadata.outer_strides,
    )) {
        *slot = sum_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len));
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn sum_axis_strided<T: Numeric>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let mut reduced = vec![T::zero(); metadata.output_len];

    for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
        base_offset,
        &metadata.output_shape,
        &metadata.outer_strides,
    )) {
        *slot = sum_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn prod_axis_contiguous<T: Numeric>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let mut reduced = vec![T::zero(); metadata.output_len];

    for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
        base_offset,
        &metadata.output_shape,
        &metadata.outer_strides,
    )) {
        *slot = prod_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len));
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn prod_axis_strided<T: Numeric>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>> {
    let mut reduced = vec![T::zero(); metadata.output_len];

    for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
        base_offset,
        &metadata.output_shape,
        &metadata.outer_strides,
    )) {
        *slot = prod_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn min_axis_contiguous<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let mut reduced = vec![T::zero(); metadata.output_len];

    for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
        base_offset,
        &metadata.output_shape,
        &metadata.outer_strides,
    )) {
        *slot = min_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len), "min")?;
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn min_axis_strided<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let mut reduced = vec![T::zero(); metadata.output_len];

    for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
        base_offset,
        &metadata.output_shape,
        &metadata.outer_strides,
    )) {
        *slot = min_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn max_axis_contiguous<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let mut reduced = vec![T::zero(); metadata.output_len];

    for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
        base_offset,
        &metadata.output_shape,
        &metadata.outer_strides,
    )) {
        *slot = max_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len), "max")?;
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn max_axis_strided<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<T>>
where
    T: Numeric + PartialOrd,
{
    let mut reduced = vec![T::zero(); metadata.output_len];

    for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
        base_offset,
        &metadata.output_shape,
        &metadata.outer_strides,
    )) {
        *slot = max_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride);
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn mean_axis_contiguous<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
{
    let mut reduced = vec![0.0_f64; metadata.output_len];

    for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
        base_offset,
        &metadata.output_shape,
        &metadata.outer_strides,
    )) {
        *slot = mean_contiguous(contiguous_lane(data, lane_offset, metadata.axis_len), "mean")?;
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn mean_axis_strided<T>(
    data: &[T],
    base_offset: usize,
    metadata: AxisReductionMetadata,
) -> AtlasNdResult<NDArray<f64>>
where
    T: Numeric + ToPrimitive,
{
    let mut reduced = vec![0.0_f64; metadata.output_len];

    for (slot, lane_offset) in reduced.iter_mut().zip(offset_iter(
        base_offset,
        &metadata.output_shape,
        &metadata.outer_strides,
    )) {
        *slot =
            mean_strided_lane(data, lane_offset, metadata.axis_len, metadata.axis_stride, "mean")?;
    }

    NDArray::from_shape_vec(metadata.output_shape, reduced)
}

fn contiguous_lane<T>(data: &[T], lane_offset: usize, axis_len: usize) -> &[T] {
    &data[lane_offset..lane_offset + axis_len]
}

fn contiguous_region<T>(data: &[T], base_offset: usize, len: usize) -> &[T] {
    &data[base_offset..base_offset + len]
}

fn sum_strided_lane<T: Numeric>(
    data: &[T],
    lane_offset: usize,
    axis_len: usize,
    axis_stride: usize,
) -> T {
    let mut total = T::zero();
    let mut offset = lane_offset;

    for _ in 0..axis_len {
        total += data[offset];
        offset += axis_stride;
    }

    total
}

fn prod_strided_lane<T: Numeric>(
    data: &[T],
    lane_offset: usize,
    axis_len: usize,
    axis_stride: usize,
) -> T {
    let mut total = T::one();
    let mut offset = lane_offset;

    for _ in 0..axis_len {
        total *= data[offset];
        offset += axis_stride;
    }

    total
}

fn min_strided_lane<T>(data: &[T], lane_offset: usize, axis_len: usize, axis_stride: usize) -> T
where
    T: Numeric + PartialOrd,
{
    let mut offset = lane_offset;
    let mut current = data[offset];
    offset += axis_stride;

    for _ in 1..axis_len {
        let value = data[offset];
        if value < current {
            current = value;
        }
        offset += axis_stride;
    }

    current
}

fn max_strided_lane<T>(data: &[T], lane_offset: usize, axis_len: usize, axis_stride: usize) -> T
where
    T: Numeric + PartialOrd,
{
    let mut offset = lane_offset;
    let mut current = data[offset];
    offset += axis_stride;

    for _ in 1..axis_len {
        let value = data[offset];
        if value > current {
            current = value;
        }
        offset += axis_stride;
    }

    current
}

fn mean_strided_lane<T>(
    data: &[T],
    lane_offset: usize,
    axis_len: usize,
    axis_stride: usize,
    op: &'static str,
) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
{
    let mut total = 0.0_f64;
    let mut offset = lane_offset;

    for _ in 0..axis_len {
        total += data[offset].to_f64().ok_or(AtlasNdError::NumericConversionFailed { op })?;
        offset += axis_stride;
    }

    Ok(total / axis_len as f64)
}

fn sum_all<T: Numeric>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> T {
    let mut total = T::zero();
    for_each_value(data, offset, shape, strides, |value| {
        total += *value;
    });
    total
}

fn prod_all<T: Numeric>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> T {
    let mut total = T::one();
    for_each_value(data, offset, shape, strides, |value| {
        total *= *value;
    });
    total
}

fn min_all<T>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut minimum = None;

    for_each_value(data, offset, shape, strides, |value| {
        minimum = Some(match minimum {
            Some(current) if current < *value => current,
            Some(_) | None => *value,
        });
    });

    minimum.ok_or(AtlasNdError::EmptyReduction { op: "min" })
}

fn max_all<T>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> AtlasNdResult<T>
where
    T: Numeric + PartialOrd,
{
    let mut maximum = None;

    for_each_value(data, offset, shape, strides, |value| {
        maximum = Some(match maximum {
            Some(current) if current > *value => current,
            Some(_) | None => *value,
        });
    });

    maximum.ok_or(AtlasNdError::EmptyReduction { op: "max" })
}

fn mean_all<T>(data: &[T], offset: usize, shape: &[usize], strides: &[usize]) -> AtlasNdResult<f64>
where
    T: Numeric + ToPrimitive,
{
    let len = element_count(shape);
    if len == 0 {
        return Err(AtlasNdError::EmptyReduction { op: "mean" });
    }

    let mut total = 0.0_f64;
    try_for_each_value(data, offset, shape, strides, |value| {
        total += value.to_f64().ok_or(AtlasNdError::NumericConversionFailed { op: "mean" })?;
        Ok(())
    })?;

    Ok(total / len as f64)
}
#[cfg(test)]
mod tests {
    use crate::{AtlasNdError, NDArray};

    #[test]
    fn whole_array_reductions_work_for_contiguous_arrays() {
        let array = NDArray::from_vec(vec![2, 2], vec![1_i32, 2, 3, 4]).unwrap();

        assert_eq!(array.sum(), 10);
        assert_eq!(array.prod(), 24);
        assert_eq!(array.min().unwrap(), 1);
        assert_eq!(array.max().unwrap(), 4);
        assert_eq!(array.mean().unwrap(), 2.5);
    }

    #[test]
    fn whole_array_reductions_work_for_scalar_arrays() {
        let array = NDArray::from_shape_vec([], vec![7_i32]).unwrap();

        assert_eq!(array.sum(), 7);
        assert_eq!(array.prod(), 7);
        assert_eq!(array.min().unwrap(), 7);
        assert_eq!(array.max().unwrap(), 7);
        assert_eq!(array.mean().unwrap(), 7.0);
    }

    #[test]
    fn whole_array_reductions_work_for_non_contiguous_views() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().slice([0, 1], vec![2, 2]).unwrap();

        assert_eq!(view.sum(), 12);
        assert_eq!(view.prod(), 40);
        assert_eq!(view.min().unwrap(), 1);
        assert_eq!(view.max().unwrap(), 5);
        assert_eq!(view.mean().unwrap(), 3.0);
    }

    #[test]
    fn sum_and_prod_use_identity_for_empty_arrays() {
        let array = NDArray::<i32>::new(vec![0, 3], 7);

        assert_eq!(array.sum(), 0);
        assert_eq!(array.prod(), 1);
    }

    #[test]
    fn min_max_and_mean_reject_empty_arrays() {
        let array = NDArray::<i32>::new(vec![0, 3], 7);

        assert_eq!(array.min().unwrap_err(), AtlasNdError::EmptyReduction { op: "min" });
        assert_eq!(array.max().unwrap_err(), AtlasNdError::EmptyReduction { op: "max" });
        assert_eq!(array.mean().unwrap_err(), AtlasNdError::EmptyReduction { op: "mean" });
    }

    #[test]
    fn axis_reductions_work_for_contiguous_arrays() {
        let array = NDArray::from_vec(vec![2, 3], vec![1_i32, 2, 3, 4, 5, 6]).unwrap();

        assert_eq!(array.sum_axis(0).unwrap().data(), &[5, 7, 9]);
        assert_eq!(array.sum_axis(1).unwrap().data(), &[6, 15]);
        assert_eq!(array.prod_axis(0).unwrap().data(), &[4, 10, 18]);
        assert_eq!(array.prod_axis(1).unwrap().data(), &[6, 120]);
        assert_eq!(array.min_axis(0).unwrap().data(), &[1, 2, 3]);
        assert_eq!(array.min_axis(1).unwrap().data(), &[1, 4]);
        assert_eq!(array.max_axis(1).unwrap().data(), &[3, 6]);
        assert_eq!(array.max_axis(0).unwrap().data(), &[4, 5, 6]);
        assert_eq!(array.mean_axis(0).unwrap().data(), &[2.5, 3.5, 4.5]);
        assert_eq!(array.mean_axis(1).unwrap().data(), &[2.0, 5.0]);
    }

    #[test]
    fn axis_reductions_work_for_strided_views() {
        let array = NDArray::from_vec(vec![2, 3], vec![0_i32, 1, 2, 3, 4, 5]).unwrap();
        let view = array.view().transpose();

        assert_eq!(view.sum_axis(-2).unwrap().data(), &[3, 12]);
        assert_eq!(view.sum_axis(-1).unwrap().data(), &[3, 5, 7]);
        assert_eq!(view.prod_axis(-2).unwrap().data(), &[0, 60]);
        assert_eq!(view.min_axis(-1).unwrap().data(), &[0, 1, 2]);
        assert_eq!(view.max_axis(-2).unwrap().data(), &[2, 5]);
        assert_eq!(view.mean_axis(-1).unwrap().data(), &[1.5, 2.5, 3.5]);
    }

    #[test]
    fn axis_reductions_validate_axis_bounds() {
        let array = NDArray::new(vec![2, 3], 1_i32);

        assert_eq!(array.sum_axis(2).unwrap_err(), AtlasNdError::InvalidAxis { axis: 2, ndim: 2 });
        assert_eq!(
            array.sum_axis(-3).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: -3, ndim: 2 }
        );
        assert_eq!(array.mean_axis(2).unwrap_err(), AtlasNdError::InvalidAxis { axis: 2, ndim: 2 });
        assert_eq!(
            array.max_axis(-3).unwrap_err(),
            AtlasNdError::InvalidAxis { axis: -3, ndim: 2 }
        );
    }

    #[test]
    fn axis_reductions_handle_empty_axes_consistently() {
        let array = NDArray::<i32>::new(vec![0, 3], 1);

        assert_eq!(array.sum_axis(0).unwrap().shape(), &[3]);
        assert_eq!(array.sum_axis(0).unwrap().data(), &[0, 0, 0]);
        assert_eq!(array.prod_axis(0).unwrap().data(), &[1, 1, 1]);
        assert_eq!(array.min_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "min" });
        assert_eq!(array.max_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "max" });
        assert_eq!(array.mean_axis(0).unwrap_err(), AtlasNdError::EmptyReduction { op: "mean" });
    }

    #[test]
    fn axis_reductions_preserve_zero_length_output_shapes_when_lanes_are_empty() {
        let array = NDArray::<i32>::new(vec![2, 0, 3], 1);

        assert_eq!(array.sum_axis(0).unwrap().shape(), &[0, 3]);
        assert!(array.sum_axis(0).unwrap().data().is_empty());
        assert_eq!(array.prod_axis(2).unwrap().shape(), &[2, 0]);
        assert!(array.prod_axis(2).unwrap().data().is_empty());
        assert_eq!(array.min_axis(2).unwrap().shape(), &[2, 0]);
        assert!(array.min_axis(2).unwrap().data().is_empty());
        assert_eq!(array.max_axis(0).unwrap().shape(), &[0, 3]);
        assert!(array.max_axis(0).unwrap().data().is_empty());
        assert_eq!(array.mean_axis(2).unwrap().shape(), &[2, 0]);
        assert!(array.mean_axis(2).unwrap().data().is_empty());
    }

    #[test]
    fn axis_reductions_return_scalar_outputs_for_one_dimensional_inputs() {
        let array = NDArray::from_shape_vec([4], vec![1_i32, 2, 3, 4]).unwrap();

        assert_eq!(array.sum_axis(-1).unwrap().shape(), &[] as &[usize]);
        assert_eq!(array.sum_axis(-1).unwrap().data(), &[10]);
        assert_eq!(array.prod_axis(-1).unwrap().data(), &[24]);
        assert_eq!(array.min_axis(-1).unwrap().data(), &[1]);
        assert_eq!(array.max_axis(-1).unwrap().data(), &[4]);
        assert_eq!(array.mean_axis(-1).unwrap().data(), &[2.5]);
    }
}
