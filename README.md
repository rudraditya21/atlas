# Atlas

Atlas is an early-stage, CPU-first numerical computing workspace for Rust. It provides dense
arrays and views, linear algebra, descriptive statistics, reproducible random sampling,
classical machine-learning baselines, and Arrow/Polars interoperability.

## Crates

| Crate | Purpose |
| --- | --- |
| `atlas-ndarray` | Dense n-dimensional arrays, logical views, slicing, broadcasting, elementwise operations, and reductions. |
| `atlas-linalg` | Dense and batched linear algebra, factorizations, iterative solves, and symmetric eigendecomposition. |
| `atlas-stats` | Descriptive, weighted, axis-wise, covariance, correlation, and quantile statistics. |
| `atlas-random` | Seeded random generation, distributions, shuffling, and sampling. |
| `atlas-ml` | Classical ML baselines, preprocessing, evaluation, splitting, and KNN search backends. |
| `atlas-arrow` | Safe conversion between Atlas arrays and Apache Arrow primitives and record batches. |
| `atlas-polars` | Safe conversion between Atlas arrays and Polars series and data frames. |
| `atlas-constants` | Mathematical constants for `f32` and `f64`. |

## Quick start

Atlas currently uses workspace path dependencies. Add the crate you need to a package within this
workspace, then use its public API:

```rust
use atlas_ndarray::NDArray;

let values = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
let doubled = values.mul(2.0);

assert_eq!(doubled.data(), &[2.0, 4.0, 6.0, 8.0]);
```

## Development

Atlas requires Rust 1.85 or newer.

```sh
cargo build --workspace --locked
cargo test --workspace --locked
cargo bench --bench ndarray_contiguous_kernels
```

### Formatting

Atlas uses nightly `rustfmt` to group Rust imports. Install it once before using `make format`:

```sh
rustup toolchain install nightly --component rustfmt
make format
```

### Python extension

The Python extension requires Python 3.10 or newer, Rust 1.85 or newer, and
[uv](https://docs.astral.sh/uv/).

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv sync
make python-dev
```

Activate the environment when running Python commands directly:

```sh
source .venv/bin/activate
```

Run the Python test suite with its test dependencies and private native test support:

```sh
make python-test
```

`make python-dev` installs the complete default extension; it never enables test-only features.
`make python-test` enables the private `test-support` feature solely for native error-translation
tests, then runs pytest.

Equivalent manual test workflow:

```sh
uv sync --extra test
uv run --with maturin maturin develop --features test-support
uv run pytest python/tests
```

## Status

Atlas is pre-1.0 software. APIs, performance characteristics, and crate publication status may
change while the library is being strengthened.
