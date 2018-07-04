# Indexed BitVec [![Build Status](https://travis-ci.org/DarkOtter/indexed-bitvec-rs.svg?branch=master)](https://travis-ci.org/DarkOtter/indexed-bitvec-rs)

This library provides an indexing system for bitvectors which should hopefully
allow fast rank and select operations. The library has not yet been completed,
tested, measured for speed or optimised so it's not yet certain this will
work.

This library is based on the design proposed by Zhou, Andersen and Kaminsky in
*Space–Efficient, High–Performance Rank & Select Structures on Uncompressed Bit Sequences*.

## Timings

Timings were made:
- Using criterion for the timing
- For a 1GB vector (slightly larger in fact)
- Compiled in release mode
- With LTO, codegen-units 1, non-incremental build
- With RUSTFLAGS='-C target-cpu=native'
- CPU: Intel® Core™ i7-7700K CPU @ 4.20GHz

| Operation   | Time   |
|-------------|--------|
| build_index | ~200ms |
| count       | ~2ns   |
| rank        | ~55ns  |
| select      | ~110ns |

The use of target-cpu is likely to have a significant effect on the speed of
the operations as it allows llvm & rust to use vectorisation & popcount
instructions that might be available, so consider working out how to compile
for the specific cpu you want to run the program on.

The timing of building the index has not been left in the code, as criterion
does so many iterations to time it that it makes running the benchmarks
take 15 minutes.

## See also

I think there is an implementation of the same approach in a
[Haskell succinct vector library](https://github.com/Gabriel439/Haskell-Succinct-Vector-Library/blob/03fb94757b68b990664f3e0ce7ea69c7c1c15ca3/src/Succinct/Vector/Index.hs).

Zhou, Andersen and Kaminsky. [*Space–Efficient, High–Performance Rank & Select Structures on Uncompressed Bit Sequences*](https://www.cs.cmu.edu/~./dga/papers/zhou-sea2013.pdf)
