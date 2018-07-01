/*
   Copyright 2018 DarkOtter

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
//! Type to represent bits, and basic count/rank/select functions for it.
use super::ceil_div_u64;
use bytes;
use ones_or_zeros::OnesOrZeros;
use std::cmp::min;
use std::ops::Deref;

/// Represents bits stored as a sequence of bytes (most significant bit first).
#[derive(Copy, Clone, Debug)]
pub struct Bits<T: Deref<Target = [u8]>>((T, u64));

fn big_enough(bytes: &[u8], used_bits: u64) -> bool {
    ceil_div_u64(used_bits, 8) <= bytes.len() as u64
}

impl<T: Deref<Target = [u8]>> Bits<T> {
    pub fn from(bytes: T, used_bits: u64) -> Option<Self> {
        if big_enough(bytes.deref(), used_bits) {
            Some(Bits((bytes, used_bits)))
        } else {
            None
        }
    }

    /// All of the bytes stored in the byte sequence: not just the ones actually used.
    pub fn all_bytes(&self) -> &[u8] {
        (self.0).0.deref()
    }

    /// The number of bits used in the storage.
    pub fn used_bits(&self) -> u64 {
        (self.0).1
    }

    /// The used bytes of the byte sequence: bear in mind some of the bits in the
    /// last byte may be unused.
    pub fn bytes(&self) -> &[u8] {
        let all_bytes = self.all_bytes();
        debug_assert!(big_enough(all_bytes, self.used_bits()));
        // This will not overflow because we checked the size
        // when we made self...
        &all_bytes[..ceil_div_u64(self.used_bits(), 8) as usize]
    }

    /// Deconstruct the bits storage to get back what it was constructed from.
    pub fn decompose(self) -> (T, u64) {
        self.0
    }

    /// Get the byte at a specific index.
    ///
    /// Returns `None` for out-of-bounds.
    ///
    /// ```
    /// use indexed_bitvec_core::*;
    /// let bits = Bits::from(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.get(0), Some(true));
    /// assert_eq!(bits.get(7), Some(false));
    /// assert_eq!(bits.get(14), Some(true));
    /// assert_eq!(bits.get(15), None);
    /// ```
    pub fn get(&self, idx_bits: u64) -> Option<bool> {
        if idx_bits >= self.used_bits() {
            None
        } else {
            debug_assert!(big_enough(self.all_bytes(), self.used_bits()));
            Some(bytes::get_unchecked(self.all_bytes(), idx_bits))
        }
    }

    /// Count the set/unset bits (*O(n)*).
    ///
    /// ```
    /// use indexed_bitvec_core::*;
    /// let bits = Bits::from(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.count::<OneBits>(), 14);
    /// assert_eq!(bits.count::<ZeroBits>(), 1);
    /// assert_eq!(bits.count::<OneBits>() + bits.count::<ZeroBits>(), bits.used_bits());
    /// ```
    pub fn count<W: OnesOrZeros>(&self) -> u64 {
        if (self.used_bits() % 8) != 0 {
            bytes::rank::<W>(self.all_bytes(), self.used_bits())
                .expect("Internally called rank out-of-range")
        } else {
            bytes::count::<W>(self.bytes())
        }
    }

    /// Count the set/unset bits before a position in the bits (*O(n)*).
    ///
    /// Returns `None` it the index is out of bounds.
    ///
    /// ```
    /// use indexed_bitvec_core::*;
    /// let bits = Bits::from(vec![0xFE, 0xFE], 15).unwrap();
    /// assert!((0..bits.used_bits()).all(|idx|
    ///     bits.rank::<OneBits>(idx).unwrap()
    ///     + bits.rank::<ZeroBits>(idx).unwrap()
    ///     == (idx as u64)));
    /// assert_eq!(bits.rank::<OneBits>(7), Some(7));
    /// assert_eq!(bits.rank::<ZeroBits>(7), Some(0));
    /// assert_eq!(bits.rank::<OneBits>(8), Some(7));
    /// assert_eq!(bits.rank::<ZeroBits>(8), Some(1));
    /// assert_eq!(bits.rank::<OneBits>(9), Some(8));
    /// assert_eq!(bits.rank::<ZeroBits>(9), Some(1));
    /// assert_eq!(bits.rank::<OneBits>(15), None);
    /// ```
    pub fn rank<W: OnesOrZeros>(&self, idx: u64) -> Option<u64> {
        if idx >= self.used_bits() {
            None
        } else {
            Some(
                bytes::rank::<W>(self.all_bytes(), idx)
                    .expect("Internally called rank out-of-range"),
            )
        }
    }

    /// Find the position of a bit by its rank (*O(n)*).
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank::<W>(result) == target_rank`
    /// and `get(result) == Some(W::is_ones())`.
    ///
    /// ```
    /// use indexed_bitvec_core::*;
    /// let bits = Bits::from(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.select::<OneBits>(6), Some(6));
    /// assert_eq!(bits.select::<OneBits>(7), Some(8));
    /// assert_eq!(bits.select::<ZeroBits>(0), Some(7));
    /// assert_eq!(bits.select::<ZeroBits>(1), None);
    /// ```
    pub fn select<W: OnesOrZeros>(&self, target_rank: u64) -> Option<u64> {
        let res = bytes::select::<W>(self.bytes(), target_rank);
        match res {
            None => None,
            Some(res) => {
                if res >= self.used_bits() {
                    None
                } else {
                    Some(res)
                }
            }
        }
    }

    /// Split the bits into a sequence of chunks of up to *n* bytes.
    pub fn chunks_by_bytes(&self, bytes_per_chunk: usize) -> impl Iterator<Item = Bits<&[u8]>> {
        let available_bits = self.used_bits();
        let bits_per_chunk = (bytes_per_chunk as u64) * 8;
        self.bytes()
            .chunks(bytes_per_chunk)
            .enumerate()
            .map(move |(i, chunk)| {
                let used_bits = i as u64 * bits_per_chunk;
                let bits = min(available_bits - used_bits, bits_per_chunk);
                Bits::from(chunk, bits).expect("Size invariant violated")
            })
    }

    /// Drop the first *n* bytes of bits from the front of the sequence.
    ///
    /// ```
    /// use indexed_bitvec_core::*;
    /// let bits = Bits::from(vec![0xFF, 0x00], 16).unwrap();
    /// assert_eq!(bits.get(0), Some(true));
    /// assert_eq!(bits.get(8), Some(false));
    /// assert_eq!(bits.drop_bytes(1).get(0), Some(false));
    /// assert_eq!(bits.drop_bytes(1).get(8), None);
    /// ```
    pub fn drop_bytes(&self, n_bytes: usize) -> Bits<&[u8]> {
        let bytes = self.bytes();
        if n_bytes >= bytes.len() {
            panic!("Index out of bounds: tried to drop all of the bits");
        }
        Bits::from(
            &bytes[n_bytes..],
            self.used_bits() - (n_bytes as u64 * 8),
        ).expect("Checked sufficient bytes are present")
    }

    /// Create a reference to these same bits.
    pub fn clone_ref(&self) -> Bits<&[u8]> {
        Bits::from(self.all_bytes(), self.used_bits()).expect("Previously checked")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ones_or_zeros::{OneBits, ZeroBits};
    use quickcheck;
    use quickcheck::Arbitrary;

    impl Arbitrary for Bits<Box<[u8]>> {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            let data = <Vec<u8>>::arbitrary(g);
            let all_bits = data.len() as u64 * 8;
            let overflow = g.gen_range(0, 64);
            Self::from(data.into_boxed_slice(), all_bits.saturating_sub(overflow))
                .expect("Generated bits must be valid")
        }
    }

    #[test]
    fn test_get() {
        let pattern_a = [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];
        let bits_a = Bits::from(&pattern_a[..], 8 * 8).unwrap();
        for i in 0..bits_a.used_bits() {
            assert_eq!(
                bits_a.get(i).unwrap(),
                i / 8 == i % 8,
                "Differed at position {}",
                i
            )
        }

        let pattern_b = [0xff, 0xc0];
        let bits_b = Bits::from(&pattern_b[..], 10).unwrap();
        for i in 0..10 {
            assert_eq!(bits_b.get(i), Some(true), "Differed at position {}", i)
        }
        for i in 10..16 {
            assert_eq!(bits_b.get(i), None, "Differed at position {}", i)
        }
    }

    #[test]
    fn test_count() {
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| Bits::from(bytes_a, len).expect("valid");
        assert_eq!(12, make(16).count::<OneBits>());
        assert_eq!(4, make(16).count::<ZeroBits>());
        assert_eq!(12, make(15).count::<OneBits>());
        assert_eq!(3, make(15).count::<ZeroBits>());
        assert_eq!(11, make(14).count::<OneBits>());
        assert_eq!(3, make(14).count::<ZeroBits>());
        assert_eq!(11, make(13).count::<OneBits>());
        assert_eq!(2, make(13).count::<ZeroBits>());
        assert_eq!(10, make(12).count::<OneBits>());
        assert_eq!(2, make(12).count::<ZeroBits>());
        assert_eq!(10, make(11).count::<OneBits>());
        assert_eq!(1, make(11).count::<ZeroBits>());
        assert_eq!(9, make(10).count::<OneBits>());
        assert_eq!(1, make(10).count::<ZeroBits>());
        assert_eq!(9, make(9).count::<OneBits>());
        assert_eq!(0, make(9).count::<ZeroBits>());
        assert_eq!(8, make(8).count::<OneBits>());
        assert_eq!(0, make(8).count::<ZeroBits>());
        assert_eq!(7, make(7).count::<OneBits>());
        assert_eq!(0, make(7).count::<ZeroBits>());
        assert_eq!(0, make(0).count::<OneBits>());
        assert_eq!(0, make(0).count::<ZeroBits>());
    }

    #[test]
    fn test_rank() {
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| Bits::from(bytes_a, len).expect("valid");
        let bits_a = make(16);
        for i in 0..15 {
            assert_eq!(Some(make(i).count::<OneBits>()), bits_a.rank::<OneBits>(i));
            assert_eq!(
                Some(make(i).count::<ZeroBits>()),
                bits_a.rank::<ZeroBits>(i)
            );
        }
        assert_eq!(None, bits_a.rank::<OneBits>(16));
        assert_eq!(None, bits_a.rank::<ZeroBits>(16));
        assert_eq!(None, make(13).rank::<OneBits>(13));
        assert_eq!(None, make(13).rank::<ZeroBits>(13));
        assert_eq!(bits_a.rank::<OneBits>(12), make(13).rank::<OneBits>(12));
        assert_eq!(bits_a.rank::<ZeroBits>(12), make(13).rank::<ZeroBits>(12));
    }

    #[test]
    fn test_select() {
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| Bits::from(bytes_a, len).expect("valid");
        assert_eq!(Some(14), make(16).select::<OneBits>(11));
        assert_eq!(None, make(14).select::<OneBits>(11));
    }

    quickcheck! {
        fn fuzz_test(bits: Bits<Box<[u8]>>) -> () {
            let mut running_rank_ones = 0;
            let mut running_rank_zeros = 0;
            for idx in 0..bits.used_bits() {
                assert_eq!(Some(running_rank_ones), bits.rank::<OneBits>(idx));
                assert_eq!(Some(running_rank_zeros), bits.rank::<ZeroBits>(idx));
                if bits.get(idx).unwrap() {
                    assert_eq!(Some(idx), bits.select::<OneBits>(running_rank_ones));
                    running_rank_ones += 1;
                } else {
                    assert_eq!(Some(idx), bits.select::<ZeroBits>(running_rank_zeros));
                    running_rank_zeros += 1;
                }
            }
        }
    }
}
