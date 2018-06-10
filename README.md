# Indexed BitVec

This library provides an indexing system for bitvectors which should hopefully
allow fast rank and select operations. The library has not yet been completed,
tested, measured for speed or optimised so it's not yet certain this will
work.

This library is based on the design proposed by Zhou, Andersen and Kaminsky in
*Space–Efficient, High–Performance Rank & Select Structures on Uncompressed Bit Sequences*.

## See also

I think there is an implementation of the same approach in a
[Haskell succinct vector library](https://github.com/Gabriel439/Haskell-Succinct-Vector-Library/blob/03fb94757b68b990664f3e0ce7ea69c7c1c15ca3/src/Succinct/Vector/Index.hs).

Zhou, Andersen and Kaminsky. [*Space–Efficient, High–Performance Rank & Select Structures on Uncompressed Bit Sequences*](https://www.cs.cmu.edu/~./dga/papers/zhou-sea2013.pdf)
