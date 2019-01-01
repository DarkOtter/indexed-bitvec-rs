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
use crate::bytes;
use crate::ones_or_zeros::{OnesOrZeros, OneBits, ZeroBits};
use core::ops::Deref;

/// Bits stored as a sequence of bytes (most significant bit first).
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
    #[inline]
    pub fn all_bytes(&self) -> &[u8] {
        (self.0).0.deref()
    }

    /// The number of bits used in the storage.
    #[inline]
    pub fn used_bits(&self) -> u64 {
        (self.0).1
    }

    /// The used bytes of the byte sequence: bear in mind some of the bits in the
    /// last byte may be unused.
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        let all_bytes = self.all_bytes();
        debug_assert!(big_enough(all_bytes, self.used_bits()));
        // This will not overflow because we checked the size
        // when we made self...
        &all_bytes[..ceil_div_u64(self.used_bits(), 8) as usize]
    }

    /// Deconstruct the bits storage to get back what it was constructed from.
    #[inline]
    pub fn decompose(self) -> (T, u64) {
        self.0
    }

    /// Get the byte at a specific index.
    ///
    /// Returns `None` for out-of-bounds.
    ///
    /// ```
    /// use indexed_bitvec_core::bits::Bits;
    /// let bits = Bits::from(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.get(0), Some(true));
    /// assert_eq!(bits.get(7), Some(false));
    /// assert_eq!(bits.get(14), Some(true));
    /// assert_eq!(bits.get(15), None);
    /// ```
    #[inline]
    pub fn get(&self, idx_bits: u64) -> Option<bool> {
        if idx_bits >= self.used_bits() {
            None
        } else {
            debug_assert!(big_enough(self.all_bytes(), self.used_bits()));
            Some(bytes::get_unchecked(self.all_bytes(), idx_bits))
        }
    }

    /// Count the set bits (*O(n)*).
    ///
    /// ```
    /// use indexed_bitvec_core::bits::Bits;
    /// let bits = Bits::from(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.count_ones(), 14);
    /// assert_eq!(bits.count_zeros(), 1);
    /// assert_eq!(bits.count_ones() + bits.count_zeros(), bits.used_bits());
    /// ```
    pub fn count_ones(&self) -> u64 {
        if (self.used_bits() % 8) != 0 {
            bytes::rank_ones(self.all_bytes(), self.used_bits())
                .expect("Internally called rank out-of-range")
        } else {
            bytes::count_ones(self.bytes())
        }
    }

    /// Count the unset bits (*O(n)*).
    ///
    /// ```
    /// use indexed_bitvec_core::bits::Bits;
    /// let bits = Bits::from(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.count_ones(), 14);
    /// assert_eq!(bits.count_zeros(), 1);
    /// assert_eq!(bits.count_ones() + bits.count_zeros(), bits.used_bits());
    /// ```
    #[inline]
    pub fn count_zeros(&self) -> u64 {
        ZeroBits::convert_count(self.count_ones(), self.used_bits())
    }

    /// Count the set bits before a position in the bits (*O(n)*).
    ///
    /// Returns `None` it the index is out of bounds.
    ///
    /// ```
    /// use indexed_bitvec_core::bits::Bits;
    /// let bits = Bits::from(vec![0xFE, 0xFE], 15).unwrap();
    /// assert!((0..bits.used_bits()).all(|idx|
    ///     bits.rank_ones(idx).unwrap()
    ///     + bits.rank_zeros(idx).unwrap()
    ///     == (idx as u64)));
    /// assert_eq!(bits.rank_ones(7), Some(7));
    /// assert_eq!(bits.rank_zeros(7), Some(0));
    /// assert_eq!(bits.rank_ones(8), Some(7));
    /// assert_eq!(bits.rank_zeros(8), Some(1));
    /// assert_eq!(bits.rank_ones(9), Some(8));
    /// assert_eq!(bits.rank_zeros(9), Some(1));
    /// assert_eq!(bits.rank_ones(15), None);
    /// ```
    pub fn rank_ones(&self, idx: u64) -> Option<u64> {
        if idx >= self.used_bits() {
            None
        } else {
            Some(bytes::rank_ones(self.all_bytes(), idx).expect(
                "Internally called rank out-of-range",
            ))
        }
    }

    /// Count the unset bits before a position in the bits (*O(n)*).
    ///
    /// Returns `None` it the index is out of bounds.
    ///
    /// ```
    /// use indexed_bitvec_core::bits::Bits;
    /// let bits = Bits::from(vec![0xFE, 0xFE], 15).unwrap();
    /// assert!((0..bits.used_bits()).all(|idx|
    ///     bits.rank_ones(idx).unwrap()
    ///     + bits.rank_zeros(idx).unwrap()
    ///     == (idx as u64)));
    /// assert_eq!(bits.rank_ones(7), Some(7));
    /// assert_eq!(bits.rank_zeros(7), Some(0));
    /// assert_eq!(bits.rank_ones(8), Some(7));
    /// assert_eq!(bits.rank_zeros(8), Some(1));
    /// assert_eq!(bits.rank_ones(9), Some(8));
    /// assert_eq!(bits.rank_zeros(9), Some(1));
    /// assert_eq!(bits.rank_ones(15), None);
    /// ```
    #[inline]
    pub fn rank_zeros(&self, idx: u64) -> Option<u64> {
        self.rank_ones(idx).map(|rank_ones| {
            ZeroBits::convert_count(rank_ones, idx)
        })
    }

    pub(crate) fn select<W: OnesOrZeros>(&self, target_rank: u64) -> Option<u64> {
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

    /// Find the position of a set bit by its rank (*O(n)*).
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank_ones(result) == Some(target_rank)`
    /// and `get(result) == Some(true)`.
    ///
    /// ```
    /// use indexed_bitvec_core::bits::Bits;
    /// let bits = Bits::from(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.select_ones(6), Some(6));
    /// assert_eq!(bits.select_ones(7), Some(8));
    /// assert_eq!(bits.select_zeros(0), Some(7));
    /// assert_eq!(bits.select_zeros(1), None);
    /// ```
    pub fn select_ones(&self, target_rank: u64) -> Option<u64> {
        self.select::<OneBits>(target_rank)
    }

    /// Find the position of an unset bit by its rank (*O(n)*).
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank_zeros(result) == Some(target_rank)`
    /// and `get(result) == Some(false)`.
    ///
    /// ```
    /// use indexed_bitvec_core::bits::Bits;
    /// let bits = Bits::from(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.select_ones(6), Some(6));
    /// assert_eq!(bits.select_ones(7), Some(8));
    /// assert_eq!(bits.select_zeros(0), Some(7));
    /// assert_eq!(bits.select_zeros(1), None);
    /// ```
    pub fn select_zeros(&self, target_rank: u64) -> Option<u64> {
        self.select::<ZeroBits>(target_rank)
    }

    /// Create a reference to these same bits.
    pub fn clone_ref(&self) -> Bits<&[u8]> {
        Bits::from(self.all_bytes(), self.used_bits()).expect("Previously checked")
    }
}

impl<T: core::ops::DerefMut<Target = [u8]>> Bits<T> {
    #[inline]
    pub fn all_bytes_mut(&mut self) -> &mut [u8] {
        (self.0).0.deref_mut()
    }

    /// Set the byte at a specific index.
    ///
    /// Returns an error if the index is out of bounds.
    ///
    /// ```
    /// use indexed_bitvec_core::bits::Bits;
    /// let mut bits = Bits::from(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.get(0), Some(true));
    /// assert_eq!(bits.get(7), Some(false));
    /// assert_eq!(bits.get(14), Some(true));
    /// assert_eq!(bits.get(15), None);
    /// assert!(bits.set(0, false).is_ok());
    /// assert_eq!(bits.get(0), Some(false));
    /// assert!(bits.set(0, true).is_ok());
    /// assert_eq!(bits.get(0), Some(true));
    /// assert!(bits.set(7, false).is_ok());
    /// assert_eq!(bits.get(7), Some(false));
    /// assert!(bits.set(14, false).is_ok());
    /// assert_eq!(bits.get(14), Some(false));
    /// assert!(bits.set(15, false).is_err());
    /// ```
    pub fn set(&mut self, idx_bits: u64, to: bool) -> Result<(), &'static str> {
        let used_bits = self.used_bits();
        if idx_bits >= used_bits {
            Err("Index out-of-bounds")
        } else {
            debug_assert!(big_enough(self.all_bytes_mut(), used_bits));
            Ok(bytes::set_unchecked(self.all_bytes_mut(), idx_bits, to))
        }
    }
}

use core::cmp::{min, Ordering, Ord};

fn must_have_or_bug<T>(opt: Option<T>) -> T {
    opt.expect(
        "If this is None there is a bug in Bits implementation")
}

fn cmp_bits(l: Bits<&[u8]>, r: Bits<&[u8]>) -> Ordering {
    let common_len = min(l.used_bits(), r.used_bits());
    let common_full_byte_len = (common_len / 8) as usize;

    let full_bytes_l = &(l.all_bytes())[..common_full_byte_len];
    let full_bytes_r = &(r.all_bytes())[..common_full_byte_len];
    match full_bytes_l.cmp(full_bytes_r) {
        Ordering::Equal => (),
        r => return r,
    };

    for idx in ((common_full_byte_len * 8) as u64)..common_len {
        let l_bit = must_have_or_bug(l.get(idx));
        let r_bit = must_have_or_bug(r.get(idx));
        match l_bit.cmp(&r_bit) {
            Ordering::Equal => (),
            r => return r,
        }
    }

    l.used_bits().cmp(&r.used_bits())
}


impl<T: Deref<Target = [u8]>> core::cmp::Ord for Bits<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_bits(self.clone_ref(), other.clone_ref())
    }
}

impl<T: Deref<Target = [u8]>> core::cmp::PartialOrd for Bits<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Deref<Target = [u8]>> core::cmp::Eq for Bits<T> {

}

impl<T: Deref<Target = [u8]>> core::cmp::PartialEq for Bits<T> {
    fn eq(&self, other: &Self) -> bool {
        self.used_bits() == other.used_bits()
            && self.cmp(other) == Ordering::Equal
    }
}

#[cfg(feature = "implement_heapsize")]
impl<T: core::ops::Deref<Target = [u8]> + heapsize::HeapSizeOf> heapsize::HeapSizeOf for Bits<T> {
    fn heap_size_of_children(&self) -> usize {
        (self.0).0.heap_size_of_children()
    }
}

#[derive(Debug)]
pub struct BitIterator<T: core::ops::Deref<Target = [u8]>> {
    search_from: u64,
    search_in: Bits<T>,
}

impl<T: core::ops::Deref<Target = [u8]>> Iterator for BitIterator<T> {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        match self.search_in.get(self.search_from) {
            None => None,
            ret => {
                self.search_from += 1;
                ret
            },
        }
    }
}

impl<T: core::ops::Deref<Target = [u8]>> BitIterator<T> {
    fn next_index<F>(&mut self, with_remaining: F) -> Option<u64>
        where F: Fn(Bits<&[u8]>, u64) -> Option<u64>
    {
        if self.search_from >= self.search_in.used_bits() {
            return None;
        }

        let byte_index = (self.search_from / 8) as usize;
        let byte_offset = self.search_from % 8;

        let byte_index_bits = (byte_index as u64) * 8;

        let remaining_part =
            must_have_or_bug(Bits::from(
                &(self.search_in.all_bytes())[byte_index..],
                self.search_in.used_bits() - byte_index_bits));

        let next_rank = with_remaining(remaining_part, byte_offset);

        match next_rank {
            None => {
                self.search_from = self.search_in.used_bits();
                None
            },
            Some(next_sub_idx) => {
                debug_assert!(next_sub_idx >= byte_offset);
                let res = byte_index_bits + next_sub_idx;
                self.search_from = res + 1;
                Some(res)
            },
        }
    }
}

#[derive(Debug)]
pub struct SetBitIndexIterator<T: core::ops::Deref<Target = [u8]>>(BitIterator<T>);

impl<T: core::ops::Deref<Target = [u8]>> Iterator for SetBitIndexIterator<T> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        self.0.next_index(|remaining_part, byte_offset| {
            debug_assert!(byte_offset < 8);
            let target_rank = must_have_or_bug(remaining_part.rank_ones(byte_offset));
            remaining_part.select_ones(target_rank)
        })
    }
}

#[derive(Debug)]
pub struct ZeroBitIndexIterator<T: core::ops::Deref<Target = [u8]>>(BitIterator<T>);

impl<T: core::ops::Deref<Target = [u8]>> Iterator for ZeroBitIndexIterator<T> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        self.0.next_index(|remaining_part, byte_offset| {
            debug_assert!(byte_offset < 8);
            let target_rank = must_have_or_bug(remaining_part.rank_zeros(byte_offset));
            remaining_part.select_zeros(target_rank)
        })
    }
}

impl<T: core::ops::Deref<Target = [u8]>> IntoIterator for Bits<T> {
    type Item = bool;
    type IntoIter = BitIterator<T>;

    fn into_iter(self) -> BitIterator<T> {
        BitIterator { search_from: 0, search_in: self }
    }
}

impl<T: core::ops::Deref<Target = [u8]>> Bits<T> {
    pub fn iter(&self) -> BitIterator<&[u8]> {
        self.clone_ref().into_iter()
    }

    pub fn into_iter_set_bits(self) -> SetBitIndexIterator<T> {
        SetBitIndexIterator(self.into_iter())
    }

    pub fn iter_set_bits(&self) -> SetBitIndexIterator<&[u8]> {
        self.clone_ref().into_iter_set_bits()
    }

    pub fn into_iter_zero_bits(self) -> ZeroBitIndexIterator<T> {
        ZeroBitIndexIterator(self.into_iter())
    }

    pub fn iter_zero_bits(&self) -> ZeroBitIndexIterator<&[u8]> {
        self.clone_ref().into_iter_zero_bits()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::boxed::Box;
    use std::vec::Vec;
    use quickcheck::Arbitrary;

    impl Arbitrary for Bits<Box<[u8]>> {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            use rand::Rng;
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
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| Bits::from(bytes_a, len).expect("valid");
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
        let make = |len: u64| Bits::from(bytes_a, len).expect("valid");
        assert_eq!(Some(14), make(16).select_ones(11));
        assert_eq!(None, make(14).select_ones(11));
    }

    quickcheck! {
        fn fuzz_test(bits: Bits<Box<[u8]>>) -> () {
            let mut running_rank_ones = 0;
            let mut running_rank_zeros = 0;
            for idx in 0..bits.used_bits() {
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

    
    impl<T: Deref<Target = [u8]>> Bits<T> {
        fn to_bool_vec_slow(&self) -> Vec<bool> {
            (0..self.used_bits())
                .map(|idx| self.get(idx).unwrap())
                .collect()
        }
    }

    quickcheck! {
        fn test_cmp_eq_model(l: Bits<Box<[u8]>>, r: Bits<Box<[u8]>>) -> () {
            let l_vec = l.to_bool_vec_slow();
            let r_vec = r.to_bool_vec_slow();
            assert_eq!(l_vec.cmp(&r_vec), l.cmp(&r));
            assert_eq!(l_vec.eq(&r_vec), l.eq(&r));
        }
    }

    #[test]
    fn test_eq_cmp() {
        fn check (expected: Ordering,
                  l: Option<Bits<Vec<u8>>>,
                  r: Option<Bits<Vec<u8>>>) {
            let l = l.unwrap();
            let r = r.unwrap();
            let expected_eq = match expected {
                Ordering::Equal => true,
                _ => false,
            };
            assert_eq!(expected_eq, l.eq(&r));
            assert_eq!(expected, l.cmp(&r));
        }

        // Should ignore extra bits
        check(Ordering::Equal,
              Bits::from(vec![0xff, 0xf0], 12),
              Bits::from(vec![0xff, 0xff], 12));

        check(Ordering::Equal,
              Bits::from(vec![], 0),
              Bits::from(vec![], 0));
        check(Ordering::Less,
              Bits::from(vec![0xff], 0),
              Bits::from(vec![0xff], 1));
        check(Ordering::Greater,
              Bits::from(vec![0xff], 1),
              Bits::from(vec![0xff], 0));
        check(Ordering::Equal,
              Bits::from(vec![0xff], 1),
              Bits::from(vec![0xff], 1));
        check(Ordering::Less,
              Bits::from(vec![0x00], 1),
              Bits::from(vec![0xff], 1));
        check(Ordering::Greater,
              Bits::from(vec![0xff], 1),
              Bits::from(vec![0x00], 1));
    }

    quickcheck! {
        fn fuzz_test_iter(bits: Bits<Box<[u8]>>) -> () {
            let as_bools: Vec<bool> =
                (0..bits.used_bits()).map(|idx| bits.get(idx).unwrap()).collect();
            let ones_locs: Vec<u64> =
                (0..bits.used_bits()).filter(|&idx| bits.get(idx).unwrap()).collect();
            let zeros_locs: Vec<u64> =
                (0..bits.used_bits()).filter(|&idx| !(bits.get(idx).unwrap())).collect();
            assert_eq!(as_bools, bits.iter().collect::<Vec<_>>());
            assert_eq!(ones_locs, bits.iter_set_bits().collect::<Vec<_>>());
            assert_eq!(zeros_locs, bits.iter_zero_bits().collect::<Vec<_>>());
        }
    }
}
