#!/bin/sh
set -e
(cd core && cargo test --verbose --no-default-features)
(cd core && cargo test --verbose --features "implement_heapsize")
cargo test --verbose --no-default-features
cargo test --verbose --features "implement_heapsize"
