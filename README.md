# Indexed Bitvector
[![Build status](https://travis-ci.org/DarkOtter/indexed-bitvec-rs.svg?branch=master)](https://travis-ci.org/DarkOtter/indexed-bitvec-rs)
[![Latest version](https://img.shields.io/crates/v/indexed_bitvec.svg)](https://crates.io/crates/indexed_bitvec)
[![Documentation](https://docs.rs/indexed_bitvec/badge.svg)](https://docs.rs/indexed_bitvec)

This library provides an indexing system for bitvectors which should hopefully
allow fast rank and select operations.

This library is based on the design proposed by Zhou, Andersen and Kaminsky in
*Space–Efficient, High–Performance Rank & Select Structures on Uncompressed Bit Sequences*.

## Licensing

This library is licensed under the GNU General Public License as published by
the Free Software Foundation, either version 3 or a later version.

If you would be interested in paying for a non-GPL commercial license please
open an issue on the [github repository](https://github.com/DarkOtter/indexed-bitvec-rs)
or otherwise contact me to discuss.

## See also

I think there is an implementation of the same approach in a
[Haskell succinct vector library](https://github.com/Gabriel439/Haskell-Succinct-Vector-Library/blob/03fb94757b68b990664f3e0ce7ea69c7c1c15ca3/src/Succinct/Vector/Index.hs).

Zhou, Andersen and Kaminsky. [*Space–Efficient, High–Performance Rank & Select Structures on Uncompressed Bit Sequences*](https://www.cs.cmu.edu/~./dga/papers/zhou-sea2013.pdf)
