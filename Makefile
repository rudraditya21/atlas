SHELL := /bin/sh

CARGO ?= cargo
UV ?= uv
WORKSPACE_FLAGS := --workspace --locked
ROOT_PACKAGE := atlas-benchmarks
BENCH_TARGETS := linalg_dense_kernels linalg_batched_kernels linalg_factorization_kernels ml_baselines ndarray_contiguous_kernels ndarray_view_kernels random_sampling_kernels stats_descriptive_kernels

.PHONY: help build check test bench bench-one python-dev python-test format clean

help:
	@printf "Available targets:\n"
	@printf "  make build           Build the full workspace with a locked dependency graph\n"
	@printf "  make check           Run fmt, check, and clippy across all workspace targets\n"
	@printf "  make test            Run the full workspace test suite with a locked dependency graph\n"
	@printf "  make bench           Run all registered root benchmark targets in stable order\n"
	@printf "  make bench-one BENCH=<name>  Run a specific benchmark target\n"
	@printf "  make python-dev      Install the default Python extension into the uv environment\n"
	@printf "  make python-test     Build test support and run the Python test suite\n"
	@printf "  make format          Format Rust imports and Python sources\n"
	@printf "  make clean           Remove build artifacts\n"

build:
	$(CARGO) build $(WORKSPACE_FLAGS)

check:
	$(CARGO) fmt --all --check
	$(CARGO) check $(WORKSPACE_FLAGS) --all-targets
	$(CARGO) clippy $(WORKSPACE_FLAGS) --all-targets -- -D warnings

test:
	$(CARGO) test $(WORKSPACE_FLAGS)

bench:
	@for bench in $(BENCH_TARGETS); do \
		$(CARGO) bench --locked --package $(ROOT_PACKAGE) --bench $$bench; \
	done

bench-one:
	@test -n "$(BENCH)" || { \
		printf "BENCH=<name> is required.\nAvailable targets: $(BENCH_TARGETS)\n"; \
		exit 1; \
	}
	@case " $(BENCH_TARGETS) " in \
		*" $(BENCH) "*) ;; \
		*) printf "Unknown bench target: %s\nAvailable targets: $(BENCH_TARGETS)\n" "$(BENCH)"; exit 1 ;; \
	esac
	$(CARGO) bench --locked --package $(ROOT_PACKAGE) --bench $(BENCH)

python-dev:
	$(UV) sync
	$(UV) run --with maturin maturin develop

python-test:
	$(UV) sync --extra test
	$(UV) run --with maturin maturin develop --features test-support
	$(UV) run pytest python/tests

format:
	$(CARGO) fmt
	rustup run nightly cargo fmt -- --config group_imports=StdExternalCrate,imports_granularity=Crate
	$(UV) run --with ruff ruff format python

clean:
	$(CARGO) clean
