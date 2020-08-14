/*
   Copyright 2020 DarkOtter

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
//!
use crate::ceil_div_u64;
use crate::ones_or_zeros::{OneBits, OnesOrZeros, ZeroBits};

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
struct AllBitsRef<'a>(&'a [u8]);

impl From<&[u8]> for AllBitsRef {
    fn from(bytes: &[u8]) -> Self {
        Self(bytes)
    }
}

impl From<AllBitsRef> for &[u8] {
    fn from(bits: BitsRef) -> Self {
        bits.0
    }
}

impl<'a> AllBitsRef<'a> {
    pub fn len_bits(self) -> u64 {
        self.0.len() as u64 * 8
    }

    unsafe fn get_unchecked(self, idx_bits: u64) -> bool {
        let data = self.0;
        let byte_idx = (idx_bits / 8) as usize;
        let idx_in_byte = (idx_bits % 8) as usize;

        debug_assert!(data.get(byte_idx).is_some());
        let byte = data.get_unchecked(byte_idx);
        let mask = 0x80 >> idx_in_byte;
        (byte & mask) != 0
    }

    pub fn count_ones(self) -> u64 {
        fn bytes_as_u64s(data: &[u8]) -> (&[u8], &[u64], &[u8]) {
            const WORD_ALIGNMENT: usize = core::mem::align_of::<u64>();
            const WORD_SIZE: usize = core::mem::size_of::<u64>();

            let total_bytes = data.len();
            if total_bytes < WORD_ALIGNMENT {
                return (data, &[], &[]);
            }

            // For actually casting we must match alignment
            let words_start = {
                let rem = (data.as_ptr() as usize) % WORD_ALIGNMENT;
                if rem != 0 {
                    WORD_ALIGNMENT - rem
                } else {
                    0
                }
            };

            debug_assert!(total_bytes.checked_sub(words_start).is_some());
            let n_words = total_bytes.wrapping_sub(words_start) / WORD_SIZE;
            debug_assert!(n_words >= 1);
            let words_end = words_start + n_words * WORD_SIZE;

            let first_range = 0..words_start;
            let words_range = words_start..words_end;
            let last_range = words_end..total_bytes;
            debug_assert!(data.get(first_range.clone()).is_some());
            debug_assert!(data.get(words_range.clone()).is_some());
            debug_assert!(data.get(last_range.clone()).is_some());
            let first_part = unsafe { data.get_unchecked(first_range) };
            let words_part = unsafe { data.get_unchecked(words_range) };
            let last_part = unsafe { data.get_unchecked(last_range) };

            let words: &[u64] = unsafe {
                use core::slice::from_raw_parts;
                debug_assert_eq!((words_part.as_ptr() as usize) % WORD_ALIGNMENT, 0);
                let ptr = words_part.as_ptr() as *const u64;
                from_raw_parts(ptr, n_words)
            };

            (first_part, words, last_part)
        }

        const MIN_SIZE_TO_SPLIT_WORDS: usize = 16 * core::mem::size_of::<u64>();

        fn count_ones_with<T: Copy, F: Fn(T) -> u32>(data: &[T], count_ones: F) -> u64 {
            data.iter().map(|&x| count_ones(x) as u64).sum::<u64>()
        }

        fn count_ones_by_bytes(data: &[u8]) -> u64 {
            count_ones_with(data, <u8>::count_ones)
        }

        fn count_ones_by_words(data: &[u64]) -> u64 {
            count_ones_with(data, <u64>::count_ones)
        }

        let data = self.0;
        if data.len() >= MIN_SIZE_TO_SPLIT_WORDS {
            let (pre_partial, words, post_partial) = bytes_as_u64s(data);
            count_ones_by_bytes(pre_partial)
                + count_ones_by_words(words)
                + count_ones_by_bytes(post_partial)
        } else {
            count_ones_by_bytes(data)
        }
    }


    pub(crate) fn select<W: OnesOrZeros>(data: &[u8], target_rank: u64) -> Option<u64> {

        /// Select a bit by rank within bytes one byte at a time, or return the total count.
        fn select_by_bytes<W: OnesOrZeros>(data: &[u8], target_rank: u64) -> Result<u64, u64> {
            let mut running_rank = 0u64;
            let mut running_index = 0u64;

            for &byte in data.iter() {
                let count = W::convert_count(byte.count_ones() as u64, 8);
                if running_rank + count > target_rank {
                    let select_in = if W::is_ones() {
                        byte as u16
                    } else {
                        (!byte) as u16
                    };
                    let selected = select_ones_u16(select_in, (target_rank - running_rank) as u32);
                    let answer = selected as u64 - 8 + running_index;
                    return Ok(answer);
                }
                running_rank += count;
                running_index += 8;
            }

            Err(running_rank)
        }

        if data.len() < MIN_SIZE_TO_SPLIT_WORDS {
            return select_by_bytes::<W>(data, target_rank).ok();
        }

        let (pre_partial, data, post_partial) = bytes_as_u64s(data);
        let pre_partial_count = match select_by_bytes::<W>(pre_partial, target_rank) {
            Ok(res) => return Some(res),
            Err(count) => count,
        };

        let mut running_rank = pre_partial_count;
        let mut running_index = pre_partial.len() as u64 * 8;

        for &word in data.iter() {
            let count = W::convert_count(word.count_ones() as u64, 64);
            if running_rank + count > target_rank {
                let answer = Word::from(u64::from_be(word))
                    .select::<W>((target_rank - running_rank) as u32)
                    .map(|sub_res| running_index + sub_res as u64);
                return answer;
            }
            running_rank += count;
            running_index += 64;
        }

        select_by_bytes::<W>(post_partial, target_rank - running_rank)
            .ok()
            .map(|sub_res| running_index + sub_res)
    }
}


#[derive(Copy, Clone, Debug)]
struct LeadingBitsRef<'a> {
    all_bits: AllBitsRef<'a>,
    skip_trailing_bits: u8,
    skipped_trailing_bits_count_ones: u8,
}

/// Bits stored as a sequence of bytes (most significant bit first).
#[derive(Copy, Clone, Debug)]
pub struct BitsRef<'a> {
    leading_bits: LeadingBitsRef<'a>,
    skip_leading_bits: u8,
    skipped_leading_bits_count_ones: u8,
}

#[inline]
fn big_enough(bytes: &[u8], idx_bits: u64) -> bool {
    (bytes.len() as u64) * 8 >= idx_bits
}

struct Positions {
    byte_range: core::ops::Range<usize>,
    leading_bits: u8,
    trailing_bits: u8,
}

impl From<core::ops::Range<u64>> for Positions {
    fn from(range: core::ops::Range<u64>) -> Self {
        let byte_pos = (range.start / 8) as usize;
        let leading_bits = (range.end % 8) as u8;
        let until = range.end;
        let byte_until = ceil_div_u64(until, 8);
        debug_assert!((byte_until * 8) >= until);
        let trailing_bits = ((byte_until * 8).wrapping_sub(until)) as u8;
        let byte_until = byte_until as usize;
        Self {
            byte_range: (byte_pos..byte_until),
            leading_bits,
            trailing_bits,
        }
    }
}

impl Positions {
    fn from_pos_len(pos: u64, len: u64) -> Self {
        Self::from(pos..pos + len)
    }
}

impl BitsRef<'static> {
    pub const fn empty() -> Self {
        BitsRef {
            data: &[],
            leading_bits: 0u8,
            trailing_bits: 0u8,
        }
    }
}

impl<'a> BitsRef<'a> {
    fn from_bytes_positions(bytes: &'a [u8], positions: Positions) -> Option<Self> {
        let Positions {
            byte_range,
            leading_bits,
            trailing_bits,
        } = positions;
        bytes.get(byte_range).map(|data| Self {
            data,
            leading_bits,
            trailing_bits,
        })
    }

    unsafe fn from_bytes_positions_unchecked(bytes: &'a [u8], positions: Positions) -> Self {
        let Positions {
            byte_range,
            leading_bits,
            trailing_bits,
        } = positions;
        let data = bytes.get_unchecked(byte_range);
        Self {
            data,
            leading_bits,
            trailing_bits,
        }
    }

    /// Get a slice of the bits in some bytes
    pub fn from_bytes(bytes: &'a [u8], pos: u64, len: u64) -> Option<Self> {
        Self::from_bytes_positions(bytes, Positions::from_pos_len(pos, len))
    }

    fn data_len_bits(&self) -> u64 {
        (self.data.len() * 8) as u64
    }

    /// The number of bits in this slice
    #[inline]
    pub fn len(&self) -> u64 {
        self.data_len_bits() as u64 - self.leading_bits as u64 - self.trailing_bits as u64
    }

    fn index_after_leading_bits(&self, idx_bits: u64) -> Option<u64> {
        let actual_idx = idx_bits + self.leading_bits as u64;
        if big_enough(self.data, actual_idx) {
            Some(actual_idx)
        } else {
            None
        }
    }

    /// Get the byte at a specific index.
    ///
    /// Returns `None` for out-of-bounds.
    #[inline]
    pub fn get(&self, idx_bits: u64) -> Option<bool> {
        self.index_after_leading_bits(idx_bits)
            .map(|actual_idx| unsafe { bytes::get_unchecked(self.data, actual_idx) })
    }

    fn count_ones_leading_bits(&self) -> u64 {
        if self.leading_bits == 0 {
            0
        } else {
            debug_assert!(!self.data.is_empty());
            let first_byte = unsafe { self.data.get_unchecked(0) };
            let mask = !(!(0u8) >> self.leading_bits);
            (mask & first_byte).count_ones() as u64
        }
    }

    unsafe fn count_ones_upto_internal(self, end_index: u64) -> u64 {
        debug_assert!(big_enough(self.data, end_index));
        let overall_count = unsafe { bytes::count_ones_upto_unchecked(self.data, end_index) };
        let leading_count = self.count_ones_leading_bits();
        debug_assert!(overall_count >= leading_count);
        overall_count.wrapping_sub(leading_count)
    }

    /// Count the set bits (*O(n)*).
    pub fn count_ones(self) -> u64 {
        unsafe { self.count_ones_upto_internal(self.data_len_bits() - self.trailing_bits as u64) }
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
        self.index_after_leading_bits(idx)
            .map(|actual_idx| unsafe { self.count_ones_upto_internal(actual_idx) })
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
        let leading_bits = self.leading_bits as u64;
        let leading_bits_count = W::convert_count(self.count_ones_leading_bits(), leading_bits);
        let inner_res = bytes::select::<W>(self.bytes(), target_rank + leading_bits_count);
        inner_res.and_then(|inner_res| {
            debug_assert!(inner_res.checked_sub(leading_bits).is_some());

        })

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

fn must_have_or_bug<T>(opt: Option<T>) -> T {
    opt.expect("If this is None there is a bug in Bits implementation")
}

fn eq_by_idx(left: &BitsRef, right: &BitsRef) -> bool {
    let left_len = left.len();
    if left_len != right.len() {
        return false;
    }
    let left_leading = left.leading_bits as u64;
    let right_leading = right.leading_bits as u64;

    (0..left_len).all(|idx| {
        let left_bit_idx = idx + left_leading;
        let right_bit_idx = idx + right_leading;
        debug_assert_eq!(left.index_after_leading_bits(idx), Some(left_bit_idx));
        debug_assert_eq!(right.index_after_leading_bits(idx), Some(right_bit_idx));
        let in_left = unsafe { bytes::get_unchecked(left.data, left_bit_idx) };
        let in_right = unsafe { bytes::get_unchecked(right.data, right_bit_idx) };
        in_left == in_right
    })
}

impl<'a> BitsRef<'a> {
    fn parts_for_byte_eq(&self) -> (&'a [u8], u8, u8, u8, u8) {
        let (leading_offset, leading_byte) = if self.leading_bits > 0 {
            (1, *unsafe { self.data.get_unchecked(0) })
        } else {
            (0, 0)
        };
        let (trailing_offset, trailing_byte) = if self.trailing_bits > 0 {
            (1, *unsafe { self.data.get_unchecked(self.data.len() - 1) })
        } else {
            (0, 0)
        };

        debug_assert!(leading_offset + trailing_offset <= self.data.len());
        let data = unsafe {
            self.data
                .get_unchecked(leading_offset..self.data.len().wrapping_sub(trailing_offset))
        };
        (
            data,
            self.leading_bits,
            leading_byte,
            self.trailing_bits,
            trailing_byte,
        )
    }
}

impl<'a> core::cmp::Eq for BitsRef<'a> {}

impl<'a> core::cmp::PartialEq for BitsRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        if self.leading_bits == other.leading_bits {
            // We can optimise by comparing byte ranges
            self.parts_for_byte_eq() == other.parts_for_byte_eq()
        } else {
            eq_by_idx(self, other)
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChunksIter<'a> {
    data: BitsRef<'a>,
    bits_in_chunk: u64,
}

impl<'a> ChunksIter<'a> {
    fn new(data: BitsRef<'a>, bits_in_chunk: u64) -> Option<Self> {
        if bits_in_chunk < 1 {
            None
        } else {
            Some(Self {
                data,
                bits_in_chunk,
            })
        }
    }
}

impl<'a> BitsRef<'a> {
    pub fn all_in_bytes(bytes: &'a [u8]) -> Self {
        Self {
            data: bytes,
            leading_bits: 0,
            trailing_bits: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
            || (self.data.len() == 1 && (self.leading_bits + self.trailing_bits) == 8)
    }

    unsafe fn split_at_after_leading_bits_unchecked(self, mid: u64) -> (Self, Self) {
        let left = self.leading_bits as u64;
        let right = self.data_len_bits() - self.trailing_bits as u64;
        debug_assert!(left <= mid && mid <= right);
        let left_pos = Positions::from(left..mid);
        let right_pos = Positions::from(mid..right);
        debug_assert!(self.data.get(left_pos.byte_range.clone()).is_some());
        debug_assert!(self.data.get(right_pos.byte_range.clone()).is_some());
        (
            Self::from_bytes_positions_unchecked(self.data, left_pos),
            Self::from_bytes_positions_unchecked(self.data, right_pos),
        )
    }

    pub fn split_at(self, split_at: u64) -> (Self, Self) {
        let mid = match self.index_after_leading_bits(split_at) {
            Some(i) => i,
            None => panic!(
                "Index out of range: split at {} on {} bits",
                split_at,
                self.len()
            ),
        };
        unsafe { self.split_at_after_leading_bits_unchecked(mid) }
    }

    fn split_at_upto(self, split_at: u64) -> (Self, Self) {
        let after_leading_bits = self.leading_bits as u64 + split_at;
        let max_mid = self.data_len_bits() - self.trailing_bits as u64;
        unsafe {
            if after_leading_bits > max_mid {
                self.split_at_after_leading_bits_unchecked(max_mid)
            } else {
                self.split_at_after_leading_bits_unchecked(after_leading_bits)
            }
        }
    }

    pub fn chunks(self, bits_in_chunk: u64) -> Option<ChunksIter<'a>> {
        ChunksIter::new(self, bits_in_chunk)
    }

    pub(crate) unsafe fn get_unchecked_slice(self, range: core::ops::Range<u64>) -> Self {
        let leading_bits = self.leading_bits as u64;
        let start = range.start + leading_bits;
        let end = range.end + leading_bits;
        let positions = Positions::from(start..end);
        Self::from_bytes_positions_unchecked(self.data, positions)
    }
}

impl<'a> ChunksIter<'a> {
    fn drop_first_n(&mut self, n: usize) {
        let (_, following_data) = self.data.split_at_upto(self.bits_in_chunk * n as u64);
        self.data = following_data;
    }
}

impl<'a> Iterator for ChunksIter<'a> {
    type Item = BitsRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let (this_chunk, following_data) = self.data.split_at_upto(self.bits_in_chunk);
        self.data = following_data;
        if this_chunk.is_empty() {
            None
        } else {
            Some(this_chunk)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }
}

impl<'a> core::iter::ExactSizeIterator for ChunksIter<'a> {
    fn len(&self) -> usize {
        ceil_div_u64(self.data.len(), self.bits_in_chunk) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::Arbitrary;
    use std::boxed::Box;
    use std::cmp::Ordering;
    use std::vec::Vec;

    fn from_or_panic<T: ?Sized + std::ops::Deref<Target = [u8]>>(bytes: &T, len: u64) -> BitsRef {
        BitsRef::from_bytes(bytes.deref(), 0u64, len)
            .expect("Tried to make an invalid BitsRef in tests")
    }

    mod gen_bits {
        use super::*;

        #[derive(Clone, Debug)]
        pub struct GenBits(Box<[u8]>, u64);

        impl GenBits {
            pub fn as_ref(&self) -> BitsRef {
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
