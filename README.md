# Atlas

Atlas is a CPU-first numerical computing library for Rust and Python. It provides dense
n-dimensional arrays and logical views, with linear algebra, statistics, random sampling,
classical machine-learning utilities, and Arrow/Polars interoperability.

Atlas is pre-1.0: public APIs and performance characteristics may change as the project
matures.

## Python

The `atlas` Python package is a native extension with a NumPy-oriented functional API. It
accepts NumPy arrays and Python sequences, and supports `bool`, signed and unsigned 8-, 16-,
32-, and 64-bit integers, plus `float32` and `float64` where an operation supports them.

```python
import numpy as np
import atlas

values = np.array([[1.0, 2.0], [3.0, 4.0]])
product = atlas.matmul(values, values)

assert atlas.mean(product) == 13.5
```

The binding includes constructors, shape and indexing operations, sorting and selection,
reductions, elementwise arithmetic, and dtype utilities. See `python/atlas/__init__.py` for the
current exported API.

### Install for development

Python development requires Python 3.10+, Rust 1.85+, and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
make python-dev
```

Run the Python suite with:

```sh
make python-test
```

`make python-test` enables private native test support; `make python-dev` builds the normal
extension only.

## Rust

The workspace crates can be used from packages in this repository:

```rust
use atlas_ndarray::NDArray;

let values = NDArray::from_shape_vec([2, 2], vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
let doubled = values.mul(2.0);

assert_eq!(doubled.data(), &[2.0, 4.0, 6.0, 8.0]);
```

| Crate | Focus |
| --- | --- |
| `atlas-ndarray` | Dense arrays, views, slicing, broadcasting, elementwise operations, and reductions. |
| `atlas-linalg` | Dense and batched linear algebra, factorizations, solves, and eigendecomposition. |
| `atlas-stats` | Descriptive, weighted, axis-wise, covariance, correlation, and quantile statistics. |
| `atlas-random` | Seeded generation, distributions, shuffling, and sampling. |
| `atlas-ml` | Classical ML baselines, preprocessing, evaluation, splitting, and KNN search. |
| `atlas-arrow`, `atlas-polars` | Interoperability with Apache Arrow and Polars. |
| `atlas-constants` | Mathematical constants for `f32` and `f64`. |

## Development

Rust development requires Rust 1.85 or newer.

```sh
make build
make check
make test
```

Formatting uses nightly `rustfmt` for grouped imports:

```sh
rustup toolchain install nightly --component rustfmt
make format
```
