#!/bin/sh
set -e
(cd indexed-bitvec-core && cargo test --verbose --no-default-features)
(cd indexed-bitvec-core && cargo test --verbose --features "implement_heapsize")
(cd indexed-bitvec && cargo test --verbose --no-default-features)
(cd indexed-bitvec && cargo test --verbose --features "implement_heapsize")
