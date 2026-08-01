.PHONY: help build check test clean

help:
	@printf "Available targets:\n"
	@printf "  make build           Build the full workspace\n"
	@printf "  make check           Run cargo check for the full workspace\n"
	@printf "  make test            Run the full workspace test suite\n"
	@printf "  make clean           Remove build artifacts\n"

build:
	cargo build --workspace

check:
	cargo check --workspace

test:
	cargo test --workspace

clean:
	cargo clean
