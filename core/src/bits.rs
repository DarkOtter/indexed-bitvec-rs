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
struct AllBits<T>(T);

impl<'a> From<&'a [u8]> for AllBits<&'a [u8]> {
    fn from(bytes: &'a [u8]) -> Self {
        Self(bytes)
    }
}

impl<'a> From<&'a mut [u8]> for AllBits<&'a mut [u8]> {
    fn from(bytes: &'a mut [u8]) -> Self {
        Self(bytes)
    }
}

fn split_idx(idx_bits: u64) -> (usize, usize) {
    ((idx_bits / 8) as usize, (idx_bits % 8) as usize)
}

const EMPTY_BYTES: &'static [u8] = &[];

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

impl<'a> AllBits<&'a [u8]> {
    pub const fn empty() -> Self {
        Self(EMPTY_BYTES)
    }

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
        use crate::word::Word;

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

    pub fn split_at_bytes(self, byte_idx: usize) -> Option<(AllBits<&'a [u8]>, AllBits<&'a [u8]>)> {
        if byte_idx > self.0.len() {
            return None;
        }
        let (l, r) = self.0.split_at(byte_idx);
        Some((Self::from(l), Self::from(r)))
    }

    pub fn split_at(self, idx_bits: u64) -> Option<(LeadingBits<&'a [u8]>, Bits<&'a [u8]>)> {
        LeadingBits::from(self, idx_bits).map(|leading_part| {
            let trailing_part = Bits::from(self.0, idx_bits, self.len().wrapping_sub(idx_bits))
                .expect("Indexes are already checked by other operation");
            (leading_part, trailing_part)
        })
    }
}

#[derive(Copy, Clone, Debug)]
struct LeadingBits<T> {
    all_bits: AllBits<T>,
    skip_trailing_bits: u8,
    skipped_trailing_bits_count_ones: u8,
}

const fn leading_from_all<T>(all_bits: AllBits<T>) -> LeadingBits<T> {
    LeadingBits {
        all_bits,
        skip_trailing_bits: 0,
        skipped_trailing_bits_count_ones: 0,
    }
}

impl<T> From<AllBits<T>> for LeadingBits<T> {
    fn from(all_bits: AllBits<T>) -> Self {
        leading_from_all(all_bits)
    }
}

fn get_partial_byte(all_bits: AllBits<&[u8]>, skip_trailing_bits: u8) -> u8 {
    let &last_byte = all_bits.0.last().expect("Bits should not be empty");
    let pow2 = 1u8 << skip_trailing_bits;
    debug_assert!(pow2.checked_sub(1).is_some());
    let mask = pow2.wrapping_sub(1);
    last_byte & mask
}

impl<'a> LeadingBits<&'a [u8]> {
    pub const fn empty() -> Self {
        leading_from_all(AllBits::empty())
    }

    pub fn from(all_bits: AllBits<&'a [u8]>, len: u64) -> Option<Self> {
        if len > all_bits.len() {
            return None;
        }
        let use_bytes = ceil_div_u64(len, 8);
        let use_bits = use_bytes * 8;
        let use_bytes = use_bytes as usize;
        debug_assert!(all_bits.0.get(..use_bytes).is_some());
        let all_bits = AllBits(unsafe { all_bits.0.get_unchecked(..use_bytes) });
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

    #[cfg(test)]
    #[inline]
    pub fn is_empty(self) -> bool {
        self.all_bits.is_empty()
    }

    pub fn get(self, idx_bits: u64) -> Option<bool> {
        if idx_bits < self.len() {
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

    pub fn split_at_bytes(
        self,
        byte_idx: usize,
    ) -> Option<(AllBits<&'a [u8]>, LeadingBits<&'a [u8]>)> {
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

    pub fn split_at(self, idx_bits: u64) -> Option<(LeadingBits<&'a [u8]>, Bits<&'a [u8]>)> {
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
                    (trailing_part.leading_bits.all_bits.0.len() as u64 * 8)
                    >= (trailing_part.leading_bits.skip_trailing_bits + trailing_part.skip_leading_bits) as u64
                );
                (leading_part, trailing_part)
            })
    }
}

/// Bits stored as a sequence of bytes (most significant bit first).
#[derive(Copy, Clone, Debug)]
pub struct Bits<T> {
    leading_bits: LeadingBits<T>,
    skip_leading_bits: u8,
    skipped_leading_bits_count_ones: u8,
}

pub type BitsRef<'a> = Bits<&'a [u8]>;

const fn bits_from_leading<T>(leading_bits: LeadingBits<T>) -> Bits<T> {
    Bits {
        leading_bits,
        skip_leading_bits: 0,
        skipped_leading_bits_count_ones: 0,
    }
}

impl<T> From<LeadingBits<T>> for Bits<T> {
    fn from(leading_bits: LeadingBits<T>) -> Self {
        bits_from_leading(leading_bits)
    }
}

impl<T> From<AllBits<T>> for Bits<T> {
    fn from(all_bits: AllBits<T>) -> Self {
        Self::from(<LeadingBits<T>>::from(all_bits))
    }
}

impl<'a> Bits<&'a [u8]> {
    pub const fn empty() -> Self {
        bits_from_leading(LeadingBits::empty())
    }

    pub fn from(bytes: &'a [u8], pos: u64, len: u64) -> Option<Self> {
        let (skip_bytes, skip_leading_bits) = split_idx(pos);
        let skip_leading_bits = skip_leading_bits as u8;
        let bytes = bytes.get(skip_bytes..)?;
        let extra_len = len + skip_leading_bits as u64;
        let leading_bits = LeadingBits::from(bytes.into(), extra_len)?;

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

    pub fn split_at(self, idx_bits: u64) -> Option<(Bits<&'a [u8]>, Bits<&'a [u8]>)> {
        // If this overflows then it must be out of range
        let actual_idx = idx_bits.checked_add(self.skip_leading_bits as u64)?;
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

    unsafe fn into_leading_bits_unchecked(self) -> LeadingBits<&'a [u8]> {
        self.leading_bits
    }

    fn into_leading_bits(self) -> Option<LeadingBits<&'a [u8]>> {
        if self.skip_leading_bits == 0 {
            Some(unsafe { self.into_leading_bits_unchecked() })
        } else {
            None
        }
    }

    pub fn chunks(self, bits_in_chunk: u64) -> Option<ChunksIter<'a>> {
        ChunksIter::new(self, bits_in_chunk)
    }
}

impl<'a> PartialEq for LeadingBits<&'a [u8]> {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        debug_assert_eq!(self.skip_trailing_bits, other.skip_trailing_bits);
        if self.skip_trailing_bits == 0 {
            self.all_bits == other.all_bits
        } else {
            let full_bytes_len = {
                let bytes_len = self.all_bits.0.len();
                debug_assert!(bytes_len.checked_sub(1).is_some());
                bytes_len.wrapping_sub(1)
            };
            let full_bytes_range = ..full_bytes_len;
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

impl<'a> Eq for LeadingBits<&'a [u8]> {}

fn generalised_eq(left: Bits<&[u8]>, right: Bits<&[u8]>) -> bool {
    unsafe fn unchecked_eq_in_range(
        left: AllBits<&[u8]>,
        right: AllBits<&[u8]>,
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
            0..left_len,
        )
    }
}

impl<'a> PartialEq for Bits<&'a [u8]> {
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

impl<'a> Eq for Bits<&'a [u8]> {}

#[derive(Debug, Clone)]
pub struct ChunksIter<'a> {
    data: Bits<&'a [u8]>,
    bits_in_chunk: u64,
}

impl<'a> ChunksIter<'a> {
    fn new(data: Bits<&'a [u8]>, bits_in_chunk: u64) -> Option<Self> {
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
    type Item = Bits<&'a [u8]>;

    fn next(&mut self) -> Option<Self::Item> {
        #[inline]
        fn take_last_chunk<'a>(iter: &mut ChunksIter<'a>) -> Option<Bits<&'a [u8]>> {
            if iter.data.is_empty() {
                None
            } else {
                Some(core::mem::replace(&mut iter.data, Bits::empty()))
            }
        }

        #[inline]
        fn use_split<'a>(
            iter: &mut ChunksIter<'a>,
            split: (Bits<&'a [u8]>, Bits<&'a [u8]>),
        ) -> Option<Bits<&'a [u8]>> {
            iter.data = split.1;
            Some(split.0)
        }

        unsafe fn whole_bytes_case<'a>(iter: &mut ChunksIter<'a>) -> Option<Bits<&'a [u8]>> {
            // Optimised case, we can use leading bytes and split bytewise
            debug_assert!(iter.data.into_leading_bits().is_some());
            let data = iter.data.into_leading_bits_unchecked();
            debug_assert_eq!(iter.bits_in_chunk % 8, 0);
            match data.split_at_bytes((iter.bits_in_chunk / 8) as usize) {
                None => take_last_chunk(iter),
                Some(split) => use_split(iter, (split.0.into(), split.1.into())),
            }
        }

        fn general_case<'a>(iter: &mut ChunksIter<'a>) -> Option<Bits<&'a [u8]>> {
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
pub mod tests {
    use super::*;
    use proptest::collection::SizeRange;
    use proptest::prelude::*;
    use std::ops::Range;
    use std::vec::Vec;

    fn same_thing<T: ?Sized>(left: &T, right: &T) -> bool {
        (left as *const T) == (right as *const T)
    }

    fn same_slice<T>(left: &[T], right: &[T]) -> bool {
        same_thing(left, right)
    }

    fn phys_equal_all_bits<T>(left: AllBits<&[T]>, right: AllBits<&[T]>) -> bool {
        let AllBits(left) = left;
        let AllBits(right) = right;
        same_slice(left, right)
    }

    fn phys_equal_leading_bits<T>(left: LeadingBits<&[T]>, right: LeadingBits<&[T]>) -> bool {
        let LeadingBits { all_bits: left_all, skip_trailing_bits: left_skip, skipped_trailing_bits_count_ones: left_count} = left;
        let LeadingBits { all_bits: right_all, skip_trailing_bits: right_skip, skipped_trailing_bits_count_ones: right_count } = right;
        phys_equal_all_bits(left_all, right_all) && left_skip == right_skip && left_count == right_count
    }

    fn phys_equal_bits<T>(left: Bits<&[T]>, right: Bits<&[T]>) -> bool {
        let Bits { leading_bits: left_leading, skip_leading_bits: left_skip, skipped_leading_bits_count_ones: left_count } = left;
        let Bits { leading_bits: right_leading, skip_leading_bits: right_skip, skipped_leading_bits_count_ones: right_count } = right;
        phys_equal_leading_bits(left_leading, right_leading) && left_skip == right_skip && left_count == right_count
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct SkipLeadingInfo {
        bytes: u8,
        bits: u8,
    }

    pub fn skip_leading_info() -> impl Strategy<Value = SkipLeadingInfo> {
        (0..8u8, 0..8u8).prop_map(|(bytes, bits)| SkipLeadingInfo { bytes, bits })
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    pub struct PreparedBits {
        data: Vec<u8>,
        skip_leading: SkipLeadingInfo,
        skip_trailing_bits: u8,
    }

    fn bytes(len_range: impl Into<SizeRange>) -> impl Strategy<Value = Vec<u8>> {
        proptest::collection::vec(any::<u8>(), len_range)
    }

    pub fn prepared_bits_with_len_range(
        len: Range<u64>,
        skip_leading: SkipLeadingInfo,
    ) -> impl Strategy<Value = PreparedBits> {
        let skip_leading_bytes = skip_leading.bytes as usize;
        let min_bytes =
            ceil_div_u64(len.start + skip_leading.bits as u64, 8) as usize + skip_leading_bytes;
        let max_bytes =
            ceil_div_u64(len.end + skip_leading.bits as u64, 8) as usize + skip_leading_bytes;
        let vec_gen = bytes(min_bytes..max_bytes);
        (vec_gen, 0..8u8).prop_map(move |(data, skip_trailing_bits)| {
            let skip_trailing_bits = skip_trailing_bits as u64;
            let len_without_trailing =
                ((data.len() - skip_leading.bytes as usize) * 8) as u64 - skip_leading.bits as u64;
            let skip_trailing_bits = if skip_trailing_bits > len_without_trailing {
                len_without_trailing
            } else {
                let len_with_trailing = len_without_trailing - skip_trailing_bits;
                if len_with_trailing > len.end {
                    len_without_trailing - len.end
                } else {
                    skip_trailing_bits
                }
            };
            let chosen_len = len_without_trailing - skip_trailing_bits;
            debug_assert!(skip_trailing_bits < 8);
            debug_assert!(len.contains(&chosen_len));
            PreparedBits {
                data,
                skip_leading,
                skip_trailing_bits: skip_trailing_bits as u8,
            }
        })
    }

    pub fn prepared_bits() -> impl Strategy<Value = PreparedBits> {
        // TODO: Up the default size
        skip_leading_info().prop_flat_map(|meta| prepared_bits_with_len_range(0..64, meta))
    }

    impl PreparedBits {
        fn all_bits(&self) -> AllBits<&[u8]> {
            AllBits::from(&self.data[self.skip_leading.bytes as usize..])
        }

        fn leading_bits(&self) -> LeadingBits<&[u8]> {
            let all_bits = self.all_bits();
            let result =
                LeadingBits::from(all_bits, all_bits.len() - self.skip_trailing_bits as u64)
                    .expect("Should be in range");
            debug_assert_eq!(result.skip_trailing_bits, self.skip_trailing_bits);
            result
        }

        pub fn bits(&self) -> Bits<&[u8]> {
            let all_bits = self.all_bits();
            let skip_leading_bits = self.skip_leading.bits as u64;
            let skip_trailing_bits = self.skip_trailing_bits as u64;
            let result = Bits::from(
                all_bits.0,
                skip_leading_bits,
                all_bits.len() - skip_leading_bits - skip_trailing_bits,
            )
            .expect("Should be in range");
            debug_assert_eq!(
                result.leading_bits.skip_trailing_bits,
                self.skip_trailing_bits
            );
            debug_assert_eq!(result.skip_leading_bits, self.skip_leading.bits);
            result
        }
    }

    #[test]
    fn test_empty() {
        let empty = Bits::empty();
        assert_eq!(empty.len(), 0);
        (0..256).for_each(|idx| {
            assert_eq!(empty.get(idx), None);
        })
    }

    #[test]
    fn test_len_basic() {
        fn check_len(bytes: &[u8], len: u64) {
            assert_eq!(AllBits::from(bytes).len(), len);
        }

        check_len(&[0], 8);
        check_len(&[0, 0], 16);
        check_len(&[0, 0, 0], 24);
        check_len(&[], 0);
        assert_eq!(AllBits::empty().len(), 0);
    }

    #[test]
    fn test_get_basic() {
        fn check_get_rule(bytes: &[u8], for_idx: impl Fn(u64) -> bool) {
            let bits = AllBits::from(bytes);
            (0..bits.len()).for_each(|idx| assert_eq!(bits.get(idx), Some(for_idx(idx))));
            (0..256).for_each(|offset| {
                assert_eq!(bits.get(bits.len() + offset), None);
            })
        }

        check_get_rule(&[0x00, 0x00, 0xff], |idx| idx >= 16);
        check_get_rule(&[0x00, 0x00, 0x7f], |idx| idx >= 17);
        check_get_rule(&[0x00, 0x00, 0x3f], |idx| idx >= 18);
        check_get_rule(&[0x00, 0x00, 0x0f], |idx| idx >= 20);
        (0..256).for_each(|idx| {
            assert_eq!(AllBits::empty().get(idx), None);
        });
    }

    fn bytes_and_poslen(len_range: Range<usize>) -> impl Strategy<Value = (Vec<u8>, u64, u64)> {
        len_range.prop_flat_map(|actual_len| {
            let vec_gen = bytes(actual_len);
            let len_bits = actual_len as u64 * 8;
            let pos_gen = 0..=len_bits;
            (vec_gen, pos_gen.clone(), pos_gen).prop_map(|(data, poslen1, poslen2)| {
                let mut poslen = [poslen1, poslen2];
                poslen.sort_unstable();
                (
                    data,
                    poslen[0],
                    poslen[1].checked_sub(poslen[0]).expect("Should be larger"),
                )
            })
        })
    }

    proptest! {
        #[test]
        fn test_from_succeeding(bytes_and_poslen in bytes_and_poslen(0..4097)) {
            let (bytes, pos, len) = bytes_and_poslen;
            let bits = AllBits::from(bytes.as_slice());
            let specific_bits = Bits::from(bytes.as_slice(), pos, len);
            assert!(specific_bits.is_some());
            let specific_bits = specific_bits.unwrap();
            assert_eq!(specific_bits.len(), len);
            (0..len).for_each(|idx| {
                assert!(specific_bits.get(idx).is_some());
                assert_eq!(specific_bits.get(idx), bits.get(idx + pos));
            });
            (0..256).for_each(|offset| {
                assert_eq!(specific_bits.get(len + offset), None);
            })
        }

        #[test]
        fn test_from_failing(bytes_and_poslen in
        (0usize..4097).prop_flat_map(|actual_len| {
            let len_bits = actual_len as u64 * 8;
            (bytes(actual_len), 0..=len_bits, 1u64..256).prop_map(move |(bytes, pos, len_overflow)| {
                (bytes, pos, (len_bits + len_overflow) - pos)
            })
        })
        ) {
            let (bytes, pos, len) = bytes_and_poslen;
            assert!(Bits::from(bytes.as_slice(), pos, len).is_none());
        }

        #[test]
        fn test_len(prepared_bits in prepared_bits()) {
            let all_bits = prepared_bits.all_bits();
            let leading_bits = prepared_bits.leading_bits();
            let bits = prepared_bits.bits();

            assert_eq!(all_bits.len().checked_sub(leading_bits.len()), Some(leading_bits.skip_trailing_bits as u64));
            assert_eq!(leading_bits.len().checked_sub(bits.len()), Some(bits.skip_leading_bits as u64));
        }

        #[test]
        fn test_is_empty_probably_not_empty(prepared_bits in prepared_bits()) {
            let all_bits = prepared_bits.all_bits();
            let leading_bits = prepared_bits.leading_bits();
            let bits = prepared_bits.bits();
            assert_eq!(all_bits.is_empty(), all_bits.len() == 0);
            assert_eq!(leading_bits.is_empty(), leading_bits.len() == 0);
            assert_eq!(bits.is_empty(), bits.len() == 0);
            if all_bits.is_empty() {
                assert!(leading_bits.is_empty());
            }
            if leading_bits.is_empty() {
                assert!(bits.is_empty());
            }
        }
    }

    #[test]
    fn test_is_empty_probably_empty() {
        assert!(AllBits::empty().is_empty());
        assert!(LeadingBits::empty().is_empty());
        assert!(Bits::empty().is_empty());

        let empty_ary = [0u8; 0];
        let empty_bits = AllBits::from(&empty_ary[..]);
        assert!(empty_bits.is_empty());
        assert!(LeadingBits::from(empty_bits, 0).unwrap().is_empty());
        assert!(Bits::from(empty_bits.0, 0, 0).unwrap().is_empty());

        let singleton_ary = [0u8; 1];
        let singleton_byte = AllBits::from(&singleton_ary[..]);
        assert!(!singleton_byte.is_empty());
        for i in 0..8 {
            assert_eq!(
                LeadingBits::from(singleton_byte, i).unwrap().is_empty(),
                i == 0
            );
            assert!(Bits::from(singleton_byte.0, i, 0).unwrap().is_empty());
        }
    }

    proptest! {
        #[test]
        fn test_get(prepared_bits in prepared_bits()) {
            let all_bits = prepared_bits.all_bits();
            let leading_bits = prepared_bits.leading_bits();
            let bits = prepared_bits.bits();

            for idx in 0..(all_bits.len() + 256) {
                if idx >= all_bits.len() {
                    assert!(all_bits.get(idx).is_none());
                    assert!(leading_bits.get(idx).is_none());
                    assert!(bits.get(idx).is_none());
                } else {
                    assert!(all_bits.get(idx).is_some());

                    if idx >= leading_bits.len() {
                        assert_ne!(leading_bits.skip_trailing_bits, 0);
                        assert!(leading_bits.get(idx).is_none());
                        assert!(bits.get(idx).is_none());
                    } else {
                        assert!(leading_bits.get(idx).is_some());
                        assert_eq!(leading_bits.get(idx), all_bits.get(idx));

                        if idx >= bits.len() {
                            assert_ne!(bits.skip_leading_bits, 0);
                            assert!(bits.get(idx).is_none());
                        } else {
                            assert!(bits.get(idx).is_some());
                            assert_eq!(bits.get(idx), leading_bits.get(idx + bits.skip_leading_bits as u64));
                        }
                    }
                }
            }
        }

        #[test]
        fn test_count(prepared_bits in prepared_bits()) {
            let all_bits = prepared_bits.all_bits();
            let leading_bits = prepared_bits.leading_bits();
            let bits = prepared_bits.bits();

            assert!(all_bits.get(all_bits.len()).is_none());
            assert!(leading_bits.get(all_bits.len()).is_none());
            assert!(bits.get(all_bits.len()).is_none());

            let mut all_counts = [0; 2];
            let mut leading_counts = [0; 2];
            let mut counts = [0; 2];

            (0..all_bits.len()).for_each(|idx| {
                all_bits.get(idx).map(|bit| all_counts[bit as usize] += 1);
                leading_bits.get(idx).map(|bit| leading_counts[bit as usize] += 1);
                bits.get(idx).map(|bit| counts[bit as usize] += 1);
            });

            assert_eq!(all_bits.count_ones(), all_counts[1]);
            assert_eq!(leading_bits.count_ones(), leading_counts[1]);
            assert_eq!([bits.count_zeros(), bits.count_ones()], counts);
        }

        #[test]
        fn test_rank(prepared_bits in prepared_bits()) {
            let all_bits = prepared_bits.all_bits();
            let leading_bits = prepared_bits.leading_bits();
            let bits = prepared_bits.bits();

            assert!(all_bits.get(all_bits.len()).is_none());
            assert!(leading_bits.get(all_bits.len()).is_none());
            assert!(bits.get(all_bits.len()).is_none());

            let mut all_counts = [0; 2];
            let mut leading_counts = [0; 2];
            let mut counts = [0; 2];

            (0..all_bits.len()).for_each(|idx| {
                let leading_rank_ones = leading_bits.rank_ones(idx);
                assert_eq!(leading_rank_ones.is_some(), leading_bits.get(idx).is_some());
                if leading_rank_ones.is_some() {
                    assert_eq!(leading_rank_ones.unwrap(), leading_counts[1]);
                }

                let rank_ones = bits.rank_ones(idx);
                let rank_zeros = bits.rank_zeros(idx);
                assert_eq!(rank_ones.is_some(), bits.get(idx).is_some());
                assert_eq!(rank_zeros.is_some(), bits.get(idx).is_some());
                if rank_ones.is_some() {
                    assert_eq!([rank_zeros.unwrap(), rank_ones.unwrap()], counts);
                }

                all_bits.get(idx).map(|bit| all_counts[bit as usize] += 1);
                leading_bits.get(idx).map(|bit| leading_counts[bit as usize] += 1);
                bits.get(idx).map(|bit| counts[bit as usize] += 1);
            });
        }

        #[test]
        fn test_select(prepared_bits in prepared_bits()) {
            let all_bits = prepared_bits.all_bits();
            let leading_bits = prepared_bits.leading_bits();
            let bits = prepared_bits.bits();

            assert!(all_bits.get(all_bits.len()).is_none());
            assert!(leading_bits.get(all_bits.len()).is_none());
            assert!(bits.get(all_bits.len()).is_none());

            let mut all_counts = [0; 2];
            let mut leading_counts = [0; 2];
            let mut counts = [0; 2];

            (0..all_bits.len()).for_each(|idx| {
                let bit = all_bits.get(idx).expect("Should be in range");
                if bit {
                    assert_eq!(all_bits.select::<OneBits>(all_counts[1]), Some(idx));
                } else {
                    assert_eq!(all_bits.select::<ZeroBits>(all_counts[0]), Some(idx));
                }

                if let Some(bit) = leading_bits.get(idx) {
                    if bit {
                        assert_eq!(leading_bits.select::<OneBits>(leading_counts[1]), Some(idx));
                    } else {
                        assert_eq!(leading_bits.select::<ZeroBits>(leading_counts[0]), Some(idx));
                    }
                }

                if let Some(bit) = bits.get(idx) {
                    if bit {
                        assert_eq!(bits.select_ones(counts[1]), Some(idx));
                    } else {
                        assert_eq!(bits.select_zeros(counts[0]), Some(idx));
                    }
                }

                all_bits.get(idx).map(|bit| all_counts[bit as usize] += 1);
                leading_bits.get(idx).map(|bit| leading_counts[bit as usize] += 1);
                bits.get(idx).map(|bit| counts[bit as usize] += 1);
            });

            assert!(all_bits.select::<OneBits>(all_counts[1]).is_none());
            assert!(all_bits.select::<ZeroBits>(all_counts[0]).is_none());
            assert!(leading_bits.select::<OneBits>(leading_counts[1]).is_none());
            assert!(leading_bits.select::<ZeroBits>(leading_counts[0]).is_none());
            assert!(bits.select_ones(counts[1]).is_none());
            assert!(bits.select_zeros(counts[0]).is_none());
        }

        #[test]
        fn test_split_at_in_range(setup in
            prepared_bits().prop_flat_map(|prepared_bits| {
                let len = prepared_bits.bits().len();
                (0..=len).prop_map(move |split_at| (prepared_bits.clone(), split_at))
            })
        ) {
            let (prepared_bits, split_at) = setup;
            let bits = prepared_bits.bits();
            let split = bits.split_at(split_at);
            assert!(split.is_some());
            let (left, right) = split.unwrap();
            assert_eq!(left.len(), split_at);
            assert_eq!(split_at + right.len(), bits.len());

            (0..split_at).for_each(|idx| {
                assert!(left.get(idx).is_some());
                assert_eq!(left.get(idx), bits.get(idx));
            });
            assert!(left.get(split_at).is_none());
            (0..(bits.len()-split_at)).for_each(|idx| {
                assert!(right.get(idx).is_some());
                assert_eq!(right.get(idx), bits.get(split_at + idx));
            });
            assert!(right.get(bits.len() - split_at).is_none());
        }

        #[test]
        fn test_split_at_out_of_range(prepared_bits in prepared_bits()) {
            let bits = prepared_bits.bits();
            (1..=256).for_each(|overflow| {
                assert!(bits.split_at(bits.len() + overflow).is_none());
            });
        }
    }

    fn slice_via_split(bits: Bits<&[u8]>, range: Range<u64>) -> Option<Bits<&[u8]>> {
        let Range { start, end } = range;
        bits.split_at(end).and_then(|(before_end, _)| {
            before_end.split_at(start).map(|(_, after_start)| after_start)
        })
    }

    /* TODO: Add testing:

    Probably work via testing len/get, then testing split on this basis, then
    using split & get to test various things.

    - test_get
    - test_count
    - test_rank
    - test_select
    - test eq


    */
}
