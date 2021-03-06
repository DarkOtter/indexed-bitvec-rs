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
//! Type to represent a reference to some bits, and basic count/rank/select functions for it.
use super::ceil_div_u64;
use crate::bytes;
use crate::ones_or_zeros::{OneBits, OnesOrZeros, ZeroBits};

/// Bits stored as a sequence of bytes (most significant bit first).
#[derive(Copy, Clone, Debug)]
pub struct BitsRef<'a>((&'a [u8], u64));

impl<'a> From<BitsRef<'a>> for (&'a [u8], u64) {
    fn from(bits: BitsRef<'a>) -> Self {
        bits.0
    }
}

fn big_enough(bytes: &[u8], len: u64) -> bool {
    ceil_div_u64(len, 8) <= bytes.len() as u64
}

impl<'a> BitsRef<'a> {
    pub fn from_bytes(bytes: &'a [u8], len: u64) -> Option<Self> {
        if big_enough(bytes, len) {
            Some(BitsRef((bytes, len)))
        } else {
            None
        }
    }

    /// All of the bytes stored in the byte sequence: not just the ones actually used.
    #[inline]
    pub fn all_bytes(self) -> &'a [u8] {
        (self.0).0
    }

    /// The number of bits used in the storage.
    #[inline]
    pub fn len(self) -> u64 {
        (self.0).1
    }

    /// The used bytes of the byte sequence: bear in mind some of the bits in the
    /// last byte may be unused.
    #[inline]
    pub fn bytes(self) -> &'a [u8] {
        let all_bytes = self.all_bytes();
        debug_assert!(big_enough(all_bytes, self.len()));
        // This will not overflow because we checked the size
        // when we made self...
        &all_bytes[..ceil_div_u64(self.len(), 8) as usize]
    }

    /// Get the byte at a specific index.
    ///
    /// Returns `None` for out-of-bounds.
    #[inline]
    pub fn get(self, idx_bits: u64) -> Option<bool> {
        if idx_bits >= self.len() {
            None
        } else {
            debug_assert!(big_enough(self.all_bytes(), self.len()));
            Some(bytes::get_unchecked(self.all_bytes(), idx_bits))
        }
    }

    /// Count the set bits (*O(n)*).
    pub fn count_ones(self) -> u64 {
        if (self.len() % 8) != 0 {
            bytes::rank_ones(self.all_bytes(), self.len())
                .expect("Internally called rank out-of-range")
        } else {
            bytes::count_ones(self.bytes())
        }
    }

    /// Count the unset bits (*O(n)*).
    #[inline]
    pub fn count_zeros(self) -> u64 {
        ZeroBits::convert_count(self.count_ones(), self.len())
    }

    /// Count the set bits before a position in the bits (*O(n)*).
    ///
    /// Returns `None` it the index is out of bounds.
    pub fn rank_ones(self, idx: u64) -> Option<u64> {
        if idx >= self.len() {
            None
        } else {
            Some(
                bytes::rank_ones(self.all_bytes(), idx)
                    .expect("Internally called rank out-of-range"),
            )
        }
    }

    /// Count the unset bits before a position in the bits (*O(n)*).
    ///
    /// Returns `None` it the index is out of bounds.
    #[inline]
    pub fn rank_zeros(self, idx: u64) -> Option<u64> {
        self.rank_ones(idx)
            .map(|rank_ones| ZeroBits::convert_count(rank_ones, idx))
    }

    pub(crate) fn select<W: OnesOrZeros>(self, target_rank: u64) -> Option<u64> {
        let res = bytes::select::<W>(self.bytes(), target_rank);
        match res {
            None => None,
            Some(res) => {
                if res >= self.len() {
                    None
                } else {
                    Some(res)
                }
            }
        }
    }

    /// Find the position of a set bit by its rank (*O(n)*).
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank_ones(result) == Some(target_rank)`
    /// and `get(result) == Some(true)`.
    pub fn select_ones(self, target_rank: u64) -> Option<u64> {
        self.select::<OneBits>(target_rank)
    }

    /// Find the position of an unset bit by its rank (*O(n)*).
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank_zeros(result) == Some(target_rank)`
    /// and `get(result) == Some(false)`.
    pub fn select_zeros(self, target_rank: u64) -> Option<u64> {
        self.select::<ZeroBits>(target_rank)
    }
}

use core::cmp::{min, Ord, Ordering};

fn must_have_or_bug<T>(opt: Option<T>) -> T {
    opt.expect("If this is None there is a bug in Bits implementation")
}

impl<'a> core::cmp::Ord for BitsRef<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        let common_len = min(self.len(), other.len());
        let common_full_byte_len = (common_len / 8) as usize;

        let full_bytes_self = &(self.all_bytes())[..common_full_byte_len];
        let full_bytes_other = &(other.all_bytes())[..common_full_byte_len];
        match full_bytes_self.cmp(full_bytes_other) {
            Ordering::Equal => (),
            r => return r,
        };

        for idx in ((common_full_byte_len * 8) as u64)..common_len {
            let self_bit = must_have_or_bug(self.get(idx));
            let other_bit = must_have_or_bug(other.get(idx));
            match self_bit.cmp(&other_bit) {
                Ordering::Equal => (),
                r => return r,
            }
        }

        self.len().cmp(&other.len())
    }
}

impl<'a> core::cmp::PartialOrd for BitsRef<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> core::cmp::Eq for BitsRef<'a> {}

impl<'a> core::cmp::PartialEq for BitsRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.cmp(other) == Ordering::Equal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::Arbitrary;
    use std::boxed::Box;
    use std::vec::Vec;

    fn from_or_panic<'a, T: ?Sized + std::ops::Deref<Target = [u8]>>(
        bytes: &'a T,
        len: u64,
    ) -> BitsRef<'a> {
        BitsRef::from_bytes(bytes.deref(), len).expect("Tried to make an invalid BitsRef in tests")
    }

    mod gen_bits {
        use super::*;

        #[derive(Clone, Debug)]
        pub struct GenBits(Box<[u8]>, u64);

        impl GenBits {
            pub fn as_ref<'a>(&'a self) -> BitsRef<'a> {
                from_or_panic(&self.0, self.1)
            }
        }

        impl Arbitrary for GenBits {
            fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
                use rand::Rng;
                let data = <Vec<u8>>::arbitrary(g);
                let all_bits = data.len() as u64 * 8;
                let overflow = g.gen_range(0, 64);
                GenBits(data.into_boxed_slice(), all_bits.saturating_sub(overflow))
            }
        }
    }
    pub use self::gen_bits::GenBits;

    #[test]
    fn test_get() {
        let pattern_a = vec![0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];
        let bits_a = from_or_panic(&pattern_a, 8 * 8);
        for i in 0..bits_a.len() {
            assert_eq!(
                bits_a.get(i).unwrap(),
                i / 8 == i % 8,
                "Differed at position {}",
                i
            )
        }

        let pattern_b = vec![0xff, 0xc0];
        let bits_b = from_or_panic(&pattern_b, 10);
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
        let make = |len: u64| from_or_panic(&bytes_a, len);
        assert_eq!(12, make(16).count_ones());
        assert_eq!(4, make(16).count_zeros());
        assert_eq!(12, make(15).count_ones());
        assert_eq!(3, make(15).count_zeros());
        assert_eq!(11, make(14).count_ones());
        assert_eq!(3, make(14).count_zeros());
        assert_eq!(11, make(13).count_ones());
        assert_eq!(2, make(13).count_zeros());
        assert_eq!(10, make(12).count_ones());
        assert_eq!(2, make(12).count_zeros());
        assert_eq!(10, make(11).count_ones());
        assert_eq!(1, make(11).count_zeros());
        assert_eq!(9, make(10).count_ones());
        assert_eq!(1, make(10).count_zeros());
        assert_eq!(9, make(9).count_ones());
        assert_eq!(0, make(9).count_zeros());
        assert_eq!(8, make(8).count_ones());
        assert_eq!(0, make(8).count_zeros());
        assert_eq!(7, make(7).count_ones());
        assert_eq!(0, make(7).count_zeros());
        assert_eq!(0, make(0).count_ones());
        assert_eq!(0, make(0).count_zeros());
    }

    #[test]
    fn test_rank() {
        let pattern_a = vec![0xff, 0xaau8];
        let make = |len: u64| from_or_panic(&pattern_a, len);
        let bits_a = make(16);
        for i in 0..15 {
            assert_eq!(Some(make(i).count_ones()), bits_a.rank_ones(i));
            assert_eq!(Some(make(i).count_zeros()), bits_a.rank_zeros(i));
        }
        assert_eq!(None, bits_a.rank_ones(16));
        assert_eq!(None, bits_a.rank_zeros(16));
        assert_eq!(None, make(13).rank_ones(13));
        assert_eq!(None, make(13).rank_zeros(13));
        assert_eq!(bits_a.rank_ones(12), make(13).rank_ones(12));
        assert_eq!(bits_a.rank_zeros(12), make(13).rank_zeros(12));
    }

    #[test]
    fn test_select() {
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| from_or_panic(&bytes_a, len);
        assert_eq!(Some(14), make(16).select_ones(11));
        assert_eq!(None, make(14).select_ones(11));
    }

    quickcheck! {
        fn fuzz_test(bits: GenBits) -> () {
            let bits = bits.as_ref();
            let mut running_rank_ones = 0;
            let mut running_rank_zeros = 0;
            for idx in 0..bits.len() {
                assert_eq!(Some(running_rank_ones), bits.rank_ones(idx));
                assert_eq!(Some(running_rank_zeros), bits.rank_zeros(idx));
                if bits.get(idx).unwrap() {
                    assert_eq!(Some(idx), bits.select_ones(running_rank_ones));
                    running_rank_ones += 1;
                } else {
                    assert_eq!(Some(idx), bits.select_zeros(running_rank_zeros));
                    running_rank_zeros += 1;
                }
            }
        }
    }

    impl<'a> BitsRef<'a> {
        fn to_bool_vec_slow(self) -> Vec<bool> {
            (0..self.len()).map(|idx| self.get(idx).unwrap()).collect()
        }
    }

    quickcheck! {
        fn test_cmp_eq_pair(l: GenBits, r: GenBits) -> () {
            let l = l.as_ref();
            let r = r.as_ref();
            let l_vec = l.to_bool_vec_slow();
            let r_vec = r.to_bool_vec_slow();
            assert_eq!(l_vec.cmp(&r_vec), l.cmp(&r));
            assert_eq!(l_vec.eq(&r_vec), l.eq(&r));
        }

        fn test_cmp_eq_single(l: GenBits) -> () {
            let l = l.as_ref();
            let r = l;
            let l_vec = l.to_bool_vec_slow();
            let r_vec = r.to_bool_vec_slow();
            assert_eq!(l_vec.cmp(&r_vec), l.cmp(&r));
            assert_eq!(l_vec.eq(&r_vec), l.eq(&r));
        }
    }

    #[test]
    fn test_eq_cmp() {
        fn check<'a>(expected: Ordering, l: BitsRef<'a>, r: BitsRef<'a>) {
            let expected_eq = match expected {
                Ordering::Equal => true,
                _ => false,
            };
            assert_eq!(expected_eq, l.eq(&r));
            assert_eq!(expected, l.cmp(&r));
        }

        // Should ignore extra bits
        check(
            Ordering::Equal,
            from_or_panic(&vec![0xff, 0xf0], 12),
            from_or_panic(&vec![0xff, 0xff], 12),
        );

        check(
            Ordering::Equal,
            from_or_panic(&vec![], 0),
            from_or_panic(&vec![], 0),
        );
        check(
            Ordering::Less,
            from_or_panic(&vec![0xff], 0),
            from_or_panic(&vec![0xff], 1),
        );
        check(
            Ordering::Greater,
            from_or_panic(&vec![0xff], 1),
            from_or_panic(&vec![0xff], 0),
        );
        check(
            Ordering::Equal,
            from_or_panic(&vec![0xff], 1),
            from_or_panic(&vec![0xff], 1),
        );
        check(
            Ordering::Less,
            from_or_panic(&vec![0x00], 1),
            from_or_panic(&vec![0xff], 1),
        );
        check(
            Ordering::Greater,
            from_or_panic(&vec![0xff], 1),
            from_or_panic(&vec![0x00], 1),
        );
    }
}

#[cfg(test)]
pub use self::tests::GenBits;
