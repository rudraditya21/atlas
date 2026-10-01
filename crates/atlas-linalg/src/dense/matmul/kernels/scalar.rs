use atlas_ndarray::Numeric;

use super::Tile;

pub(super) fn run<T: Numeric>(tile: Tile<'_, T>) {
    for row in 0..tile.rows {
        for col in 0..tile.cols {
            let mut value = tile.output[row * tile.output_stride + col];
            for k in 0..tile.inner {
                value += tile.lhs[row * tile.inner + k] * tile.rhs[k * tile.cols + col];
            }
            tile.output[row * tile.output_stride + col] = value;
        }
    }
}
