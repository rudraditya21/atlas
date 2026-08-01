.PHONY: help build check test bench bench-one clean

help:
	@printf "Available targets:\n"
	@printf "  make build           Build the full workspace\n"
	@printf "  make check           Run cargo check for the full workspace\n"
	@printf "  make test            Run the full workspace test suite\n"
	@printf "  make bench           Run all workspace-level benchmarks\n"
	@printf "  make bench-one BENCH=<name>  Run a specific benchmark target\n"
	@printf "  make clean           Remove build artifacts\n"

build:
	cargo build --workspace

check:
	cargo check --workspace

test:
	cargo test --workspace

bench:
	cargo bench

bench-one:
	cargo bench --bench $(BENCH)

clean:
	cargo clean
