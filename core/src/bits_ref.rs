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
// TODO: Make select_ones_u8
use crate::word::select_ones_u16;
use core::ops::Range;

fn add_should_not_overflow(a: u64, b: u64) -> u64 {
    debug_assert!(
        a.checked_add(b).is_some(),
        "Operation unexpectedly overflowed"
    );
    a.wrapping_add(b)
}

fn sub_should_not_overflow(a: u64, b: u64) -> u64 {
    debug_assert!(
        a.checked_sub(b).is_some(),
        "Operation unexpectedly overflowed"
    );
    a.wrapping_sub(b)
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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

fn split_idx(idx_bits: u64) -> (usize, usize) {
    ((idx_bits / 8) as usize, (idx_bits % 8) as usize)
}

impl AllBitsRef<'static> {
    pub const fn empty() -> Self {
        Self(&[])
    }
}

impl<'a> AllBitsRef<'a> {
    #[inline]
    pub fn len(self) -> u64 {
        self.0.len() as u64 * 8
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    unsafe fn get_unchecked(self, idx_bits: u64) -> bool {
        let data = self.0;
        let (byte_idx, idx_in_byte) = split_idx(idx_bits);

        let byte = data.get_unchecked(byte_idx);
        let mask = 0x80 >> idx_in_byte;
        (byte & mask) != 0
    }

    fn get(self, idx_bits: u64) -> Option<bool> {
        if idx_bits < self.len() {
            Some(unsafe { self.get_unchecked(idx_bits) })
        } else {
            None
        }
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

    fn select<W: OnesOrZeros>(self, target_rank: u64) -> Option<u64> {
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

        let data = self.0;
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

impl<'a> From<AllBitsRef<'a>> for LeadingBitsRef<'a> {
    fn from(all_bits: AllBitsRef<'a>) -> Self {
        Self {
            all_bits,
            skip_trailing_bits: 0,
            skipped_trailing_bits_count_ones: 0,
        }
    }
}

impl LeadingBitsRef<'static> {
    pub const fn empty() -> Self {
        Self {
            all_bits: AllBitsRef::empty(),
            skip_trailing_bits: 0,
            skipped_trailing_bits_count_ones: 0,
        }
    }
}

fn get_partial_byte(all_bits: AllBitsRef, skip_trailing_bits: u8) -> u8 {
    let last_byte = all_bits.0.last_byte().expect("Bits should not be empty");
    let pow2 = (1u8 << skip_trailing_bits);
    debug_assert!(pow2.checked_sub(1).is_some());
    let mask = pow2.wrapping_sub(1);
    last_byte & mask
}

impl<'a> LeadingBitsRef<'a> {
    pub fn from(all_bits: AllBitsRef<'a>, len: u64) -> Option<Self> {
        if len > all_bits.len() {
            return None;
        }
        let use_bytes = ceil_div_u64(len, 8);
        let use_bits = use_bytes * 8;
        let use_bytes = use_bytes as usize;
        debug_assert!(all_bits.0.get(..use_bytes).is_some());
        let all_bits = AllBitsRef(unsafe { all_bits.0.get_unchecked(..use_bytes) });
        debug_assert!(use_bits.checked_sub(len).map_or(false, |x| x < 8));
        let skip_trailing_bits = use_bits.wrapping_sub(len) as u8;
        let skipped_trailing_bits_count_ones = if skip_trailing_bits == 0 {
            0
        } else {
            get_partial_byte(all_bits, skip_trailing_bits).count_ones() as u8
        };
        Some(Self {
            all_bits,
            skip_trailing_bits,
            skipped_trailing_bits_count_ones,
        })
    }

    #[inline]
    pub fn len(self) -> u64 {
        sub_should_not_overflow(self.all_bits.len(), self.skip_trailing_bits as u64)
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        debug_assert!(self.skip_trailing_bits < 8);
        self.all_bits.is_empty()
    }

    pub fn get(self, idx_bits: u64) -> Option<bool> {
        if ix_bits < self.len() {
            debug_assert!(self.all_bits.get(idx_bits).is_some());
            Some(unsafe { self.all_bits.get_unchecked(idx_bits) })
        } else {
            None
        }
    }

    pub fn count_ones(self) -> u64 {
        sub_should_not_overflow(
            self.all_bits.count_ones(),
            self.skipped_trailing_bits_count_ones as u64,
        )
    }

    pub fn rank_ones(self, idx_bits: u64) -> Option<u64> {
        if idx_bits >= self.len() {
            return None;
        }
        let partial = Self::from(self.all_bits, idx_bits).expect("Already checked the index");
        Some(partial.count_ones())
    }

    fn select<W: OnesOrZeros>(self, target_rank: u64) -> Option<u64> {
        self.all_bits
            .select::<W>(target_rank)
            .filter(|&idx| idx < self.len())
    }
}

/// Bits stored as a sequence of bytes (most significant bit first).
#[derive(Copy, Clone, Debug)]
pub struct BitsRef<'a> {
    leading_bits: LeadingBitsRef<'a>,
    skip_leading_bits: u8,
    skipped_leading_bits_count_ones: u8,
}

impl<'a> From<LeadingBitsRef<'a>> for BitsRef<'a> {
    fn from(leading_bits: LeadingBitsRef<'_>) -> Self {
        Self {
            leading_bits,
            skip_leading_bits: 0,
            skipped_leading_bits_count_ones: 0,
        }
    }
}

impl BitsRef<'static> {
    pub const fn empty() -> Self {
        Self {
            leading_bits: LeadingBitsRef::empty(),
            skip_leading_bits: 0,
            skipped_leading_bits_count_ones: 0,
        }
    }
}

impl<'a> BitsRef<'a> {
    pub fn from(all_bits: AllBitsRef<'a>, pos: u64, len: u64) -> Option<Self> {
        let bytes = all_bits.0;
        let (skip_bytes, skip_leading_bits) = split_idx(pos);
        let skip_leading_bits = skip_leading_bits as u8;
        let bytes = bytes.get(skip_bytes..)?;
        let extra_len = len + skip_leading_bits as u64;
        let leading_bits = LeadingBitsRef::from(bytes.into(), extra_len)?;

        let skipped_leading_bits_count_ones = if skip_leading_bits == 0 {
            0
        } else {
            debug_assert!(leading_bits.all_bits.0.get(0).is_some());
            debug_assert!(skip_leading_bits as u64 <= leading_bits.len());
            let first_byte = unsafe { leading_bits.all_bits.0.get_unchecked(0) };
            let mask = !(!(0u8) >> skip_leading_bits);
            (mask & first_byte).count_ones() as u8
        };

        Some(Self {
            leading_bits,
            skip_leading_bits,
            skipped_leading_bits_count_ones,
        })
    }

    #[inline]
    pub fn len(self) -> u64 {
        sub_should_not_overflow(self.leading_bits.len(), self.skip_leading_bits as u64)
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn get(self, idx_bits: u64) -> Option<bool> {
        // If this overflows then it must be out of range
        let actual_idx = idx_bits.checked_add(self.skip_leading_bits as u64)?;
        self.leading_bits.get(actual_idx)
    }

    pub fn count_ones(self) -> u64 {
        sub_should_not_overflow(
            self.leading_bits.count_ones(),
            self.skipped_leading_bits_count_ones as u64,
        )
    }

    pub fn count_zeros(self) -> u64 {
        ZeroBits::convert_count(self.count_ones(), self.len())
    }

    pub fn rank_ones(self, idx_bits: u64) -> Option<u64> {
        // If this overflows then it must be out of range
        let actual_idx = idx_bits.checked_add(self.skip_leading_bits as u64)?;
        self.leading_bits
            .rank_ones(actual_idx)
            .map(|base_rank_ones| {
                sub_should_not_overflow(base_rank_ones, self.skipped_leading_bits_count_ones as u64)
            })
    }

    pub fn rank_zeros(self, idx_bits: u64) -> Option<u64> {
        self.rank_ones(idx_bits)
            .map(|rank_ones| ZeroBits::convert_count(rank_ones, idx_bits))
    }

    fn select<W: OnesOrZeros>(self, target_rank: u64) -> Option<u64> {
        let skip_leading_bits = self.skip_leading_bits as u64;
        // If this overflows then we must be out of range
        let actual_target_rank = target_rank.checked_add(W::convert_count(
            self.skipped_leading_bits_count_ones as u64,
            skip_leading_bits,
        ))?;
        self.leading_bits
            .select::<W>(actual_target_rank)
            .map(|base_select| sub_should_not_overflow(base_select, skip_leading_bits))
    }

    pub fn select_ones(self, target_rank: u64) -> Option<u64> {
        self.select::<OneBits>(target_rank)
    }

    pub fn select_zeros(self, target_rank: u64) -> Option<u64> {
        self.select::<ZeroBits>(target_rank)
    }
}

impl<'a> AllBitsRef<'a> {
    pub fn split_at_bytes(self, byte_idx: usize) -> Option<(AllBitsRef<'a>, AllBitsRef<'a>)> {
        if byte_idx > self.0.len() {
            return None;
        }
        let (l, r) = self.0.split_at(byte_idx);
        Some((Self::from(l), Self::from(r)))
    }

    pub fn split_at(self, idx_bits: u64) -> Option<(LeadingBitsRef<'a>, BitsRef<'a>)> {
        LeadingBitsRef::from(self, idx_bits).map(|leading_part| {
            let trailing_part = BitsRef::from(self, idx_bits, self.len().wrapping_sub(idx_bits))
                .expect("Indexes are already checked by other operation");
            (leading_part, trailing_part)
        })
    }
}

impl<'a> LeadingBitsRef<'a> {
    pub fn split_at_bytes(self, byte_idx: usize) -> Option<(AllBitsRef<'a>, LeadingBitsRef<'a>)> {
        let (leading_part, trailing_part) = self.all_bits.split_at_bytes(byte_idx)?;
        if trailing_part.is_empty() && self.skip_trailing_bits != 0 {
            return None;
        }
        let trailing_part = Self {
            all_bits: trailing_part,
            skip_trailing_bits: self.skip_trailing_bits,
            skipped_trailing_bits_count_ones: self.skipped_trailing_bits_count_ones,
        };
        Some((leading_part, trailing_part))
    }

    pub fn split_at(self, idx_bits: u64) -> Option<(LeadingBitsRef<'a>, BitsRef<'a>)> {
        if idx_bits > self.len() {
            return None;
        }
        self.all_bits
            .split_at(idx_bits)
            .map(|(leading_part, mut trailing_part)| {
                debug_assert_eq!(trailing_part.leading_bits.skip_trailing_bits, 0);
                debug_assert_eq!(
                    trailing_part.leading_bits.skipped_trailing_bits_count_ones,
                    0
                );
                trailing_part.leading_bits.skip_trailing_bits = self.skip_trailing_bits;
                trailing_part.leading_bits.skipped_trailing_bits_count_ones =
                    self.skipped_trailing_bits_count_ones;
                debug_assert!(
                    !(trailing_part.leading_bits.all_bits.is_empty()
                        && trailing_part.leading_bits.skip_trailing_bits != 0)
                );
                debug_assert!(
                    trailing_part.skip_leading_bits + trailing_part.leading_bits.skip_trailing_bits
                        <= 8
                );
                debug_assert!(
                    trailing_part.skip_leading_bits + trailing_part.leading_bits.skip_trailing_bits
                        == 0
                        || !trailing_part.leading_bits.all_bits.0.is_empty()
                );
                (leading_part, trailing_part)
            })
    }

    pub fn into_all_bits(self) -> Option<AllBitsRef<'a>> {
        if self.skip_trailing_bits == 0 {
            Some(self.all_bits)
        } else {
            None
        }
    }
}

impl<'a> BitsRef<'a> {
    pub fn split_at(self, idx_bits: u64) -> Option<(BitsRef<'a>, BitsRef<'a>)> {
        // If this overflows then it must be out of range
        let actual_idx = idx_bits.checked_add(self.leading_bits as u64)?;
        self.leading_bits
            .split_at(actual_idx)
            .map(|(leading_part, trailing_part)| {
                let leading_part = Self {
                    leading_bits: leading_part,
                    skip_leading_bits: self.skip_leading_bits,
                    skipped_leading_bits_count_ones: self.skipped_leading_bits_count_ones,
                };
                (leading_part, trailing_part)
            })
    }

    unsafe fn into_leading_bits_unchecked(self) -> LeadingBitsRef<'a> {
        self.leading_bits
    }

    pub fn into_leading_bits(self) -> Option<(LeadingBitsRef<'a>)> {
        if self.skip_leading_bits == 0 {
            Some(unsafe { self.into_leading_bits_unchecked() })
        } else {
            None
        }
    }
}

impl<'a> PartialEq for LeadingBitsRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        debug_assert_eq!(self.skip_trailing_bits, other.skip_trailing_bits);
        if self.skip_trailing_bits == 0 {
            self.all_bits == other.all_bits
        } else {
            let full_bytes_len = sub_should_not_overflow(self.all_bits.len(), 1);
            let full_bytes_range = (..full_bytes_len);
            let full_bytes_self = {
                debug_assert!(self.all_bits.0.get(full_bytes_range.clone()).is_some());
                unsafe { self.all_bits.0.get_unchecked(full_bytes_range.clone()) }
            };
            let full_bytes_other = {
                debug_assert!(other.all_bits.0.get(full_bytes_range.clone()).is_some());
                unsafe { other.all_bits.0.get_unchecked(full_bytes_range) }
            };
            if full_bytes_self != full_bytes_other {
                return false;
            }
            get_partial_byte(self.all_bits, self.skip_trailing_bits)
                == get_partial_byte(other.all_bits, other.skip_trailing_bits)
        }
    }
}

impl<'a> Eq for LeadingBitsRef<'a> {}

fn generalised_eq(left: BitsRef, right: BitsRef) -> bool {
    unsafe fn unchecked_eq_in_range(
        left: AllBitsRef,
        right: AllBitsRef,
        left_offset: u64,
        right_offset: u64,
        range: Range<u64>,
    ) -> bool {
        if range.end <= range.start {
            debug_assert_eq!(range.into_iter().count(), 0);
            return true;
        }

        if range.end.checked_add(left_offset).is_none() {
            panic!(
                "Will overflow when adding left_offset: offset {:?} range {:?}",
                left_offset, range
            )
        } else if range.end.checked_add(right_offset).is_none() {
            panic!(
                "Will overflow when adding right_offset: offset {:?} range {:?}",
                right_offset, range
            )
        }

        range.into_iter().all(|idx| {
            let left_idx_bits = add_should_not_overflow(idx, left_offset);
            let right_idx_bits = add_should_not_overflow(idx, right_offset);
            debug_assert!(left.get(left_idx_bits).is_some());
            debug_assert!(right.get(right_idx_bits).is_some());
            let in_left = left.get_unchecked(left_idx_bits);
            let in_right = right.get_unchecked(right_idx_bits);
            in_left == in_right
        })
    }

    let left_len = left.len();
    if left_len != right.len() {
        return false;
    }
    unsafe {
        unchecked_eq_in_range(
            left.leading_bits.all_bits,
            right.leading_bits.all_bits,
            left.skip_leading_bits as u64,
            right.skip_leading_bits as u64,
            (0..left_len),
        )
    }
}

impl<'a> PartialEq for BitsRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        if self.skip_leading_bits == other.skip_leading_bits {
            match self.into_leading_bits() {
                Some(simple_self) => {
                    let simple_other = other
                        .into_leading_bits()
                        .expect("Other one should be leading bits only too");
                    simple_self == simple_other
                }
                None => {
                    // TODO: Eventually this case could be more optimised
                    generalised_eq(self.clone(), other.clone())
                }
            }
        } else {
            generalised_eq(self.clone(), other.clone())
        }
    }
}

impl<'a> Eq for BitsRef<'a> {}

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

impl<'a> Iterator for ChunksIter<'a> {
    type Item = BitsRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        #[inline]
        fn take_last_chunk(iter: &mut ChunksIter) -> Option<BitsRef> {
            if iter.is_empty() {
                None
            } else {
                Some(core::mem::replace(&mut iter.data, BitsRef::empty()))
            }
        }

        #[inline]
        fn use_split<'a>(
            iter: &mut ChunksIter<'a>,
            split: (BitsRef<'a>, BitsRef<'a>),
        ) -> Option<BitsRef> {
            iter.data = split.1;
            Some(split.0)
        }

        unsafe fn whole_bytes_case(iter: &mut ChunksIter) -> Option<BitsRef> {
            // Optimised case, we can use leading bytes and split bytewise
            debug_assert!(iter.data.into_leading_bits().is_some());
            let data = unsafe { iter.data.into_leading_bits_unchecked() };
            debug_assert_eq!(iter.bits_in_chunk % 8, 0);
            match data.split_at_bytes((iter.bits_in_chunk / 8) as usize) {
                None => take_last_chunk(iter),
                Some(split) => use_split(iter, split.into()),
            }
        }

        fn general_case(iter: &mut ChunksIter) -> Option<BitsRef> {
            match iter.data.split_at(iter.bits_in_chunk) {
                None => take_last_chunk(iter),
                Some(split) => use_split(iter, split),
            }
        }

        let skip_leading_or_not_bytes =
            { (self.data.skip_leading_bits as u64) | (self.bits_in_chunk % 8) };
        if skip_leading_or_not_bytes == 0 {
            unsafe { whole_bytes_case(self) }
        } else {
            general_case(self)
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
    use proptest::prelude::*;
    use std::vec::Vec;

    struct PreparedBits {
        data: Vec<u8>,
        skip_leading_bytes: u8,
        skip_leading_bits: u8,
        skip_trailing_bits: u8,
    }

    fn prepared_bits() -> impl Strategy<Value = PreparedBits> {
        (0..8u8, 0..2u8).prop_flat_map(|(skip_leading_bytes, leading_trailing_bytes)| {
            let min_data_len = skip_leading_bytes as usize + leading_trailing_bytes as usize;
            let vec_gen = proptest::collection::vec(any::<u8>(), min_data_len..4096);
            let first_byte_total = if leading_trailing_bytes > 1 { 8 } else { 0 };
            let first_byte_share = (0..first_byte_total);
            let last_byte_range = (0..([0, 7, 6][leading_trailing_bytes as usize]));
            let bits_info_gen = (first_byte_share, (last_byte_range.clone(), last_byte_range));
            (vec_gen, bits_info_gen).prop_map(
                |(data, (first_byte_share, (last_byte_1, last_byte_2)))| {
                    let mut last_byte = [last_byte_1, last_byte_2];
                    last_byte.sort();
                    let last_byte_1 = last_byte[0];
                    let last_byte_2 = last_byte[1] - last_byte_1;
                    let skip_leading_bits = first_byte_share + last_byte_1;
                    let skip_trailing_bits = (first_byte_total - first_byte_share) + last_byte_2;
                    debug_assert!(skip_leading_bits >= 0 && skip_leading_bits < 8);
                    debug_assert!(skip_trailing_bits >= 0 && skip_trailing_bits < 8);
                    debug_assert_eq!(
                        ceil_div(skip_leading_bits + skip_trailing_bits, 8),
                        leading_trailing_bytes
                    );
                    PreparedBits {
                        data,
                        skip_leading_bytes,
                        skip_leading_bits,
                        skip_trailing_bits,
                    }
                },
            )
        })
    }

    impl PreparedBits {
        pub fn all_bits(&self) -> AllBitsRef {
            AllBitsRef::from(&self.data[self.skip_leading_bytes as usize..])
        }

        pub fn leading_bits(&self) -> LeadingBitsRef {
            let all_bits = self.all_bits();
            let result =
                LeadingBitsRef::from(all_bits, all_bits.len() - self.skip_trailing_bits as u64)
                    .expect("Should be in range");
            debug_assert_eq!(result.skip_trailing_bits, self.skip_trailing_bits);
            result
        }

        pub fn bits(&self) -> BitsRef {
            let all_bits = self.all_bits();
            let skip_leading_bits = self.skip_leading_bits as u64;
            let skip_trailing_bits = self.skip_trailing_bits as u64;
            let result = BitsRef::from(
                all_bits,
                skip_leading_bits,
                all_bits.len() - skip_leading_bits - skip_trailing_bits,
            )
            .expect("Should be in range");
            debug_assert_eq!(
                result.leading_bits.skip_trailing_bits,
                self.skip_trailing_bits
            );
            debug_assert_eq!(result.skip_leading_bits, self.skip_leading_bits);
            result
        }
    }

    use quickcheck::Arbitrary;
    use std::boxed::Box;
    use std::cmp::Ordering;

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
    use crate::ceil_div;

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
