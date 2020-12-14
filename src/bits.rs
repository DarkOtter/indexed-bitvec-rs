/*
 * A part of indexed-bitvec-rs, a library implementing bitvectors with fast rank operations.
 *     Copyright (C) 2020  DarkOtter
 *
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
//! Type to represent a reference to some bits, and basic count/rank/select functions for it.
//!
use crate::import::prelude::*;
use crate::word::*;

// TODO: Check everything in here is still tested
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

const EMPTY_WORDS: &'static [Word] = &[];

#[derive(Debug, Copy, Clone)]
struct LeadingBitsOf<T> {
    all_bits: T,
    skip_trailing_bits: u8,
    skipped_trailing_bits_count_ones: u8,
}

const_assert!(Word::len() <= u8::max_value() as u64);

const fn leading_from_all<T>(all_bits: T) -> LeadingBitsOf<T> {
    LeadingBitsOf {
        all_bits,
        skip_trailing_bits: 0,
        skipped_trailing_bits_count_ones: 0,
    }
}

impl<T> From<T> for LeadingBitsOf<T> {
    fn from(all_bits: T) -> Self {
        leading_from_all(all_bits)
    }
}

fn bits_len<T: Bits + ?Sized>(bits: &T) -> u64 {
    bits.len()
}

impl<'a> LeadingBitsOf<&'a [Word]> {
    pub const fn empty() -> Self {
        leading_from_all(EMPTY_WORDS)
    }
}

fn get_bit<T: Bits + ?Sized>(bits: &T, idx_bits: u64) -> Option<bool> {
    bits.get(idx_bits)
}

impl<'a> Bits for LeadingBitsOf<&'a [Word]> {
    #[inline]
    fn len(&self) -> u64 {
        sub_should_not_overflow(bits_len(self.all_bits), self.skip_trailing_bits as u64)
    }

    fn get(&self, idx_bits: u64) -> Option<bool> {
        if idx_bits >= bits_len(self) {
            return None;
        }
        let r = get_bit(self.all_bits, idx_bits);
        debug_assert!(r.is_some(), "Should not be out of bounds");
        r
    }

    fn count_ones(&self) -> u64 {
        sub_should_not_overflow(
            self.all_bits.count_ones(),
            self.skipped_trailing_bits_count_ones as u64,
        )
    }

    fn rank_ones(&self, idx_bits: u64) -> Option<u64> {
        if idx_bits >= self.len() {
            return None;
        }
        let r = self.all_bits.rank_ones(idx_bits);
        debug_assert!(r.is_some(), "Should not be out of bounds");
        r
    }

    fn select_ones(&self, target_rank: u64) -> Option<u64> {
        self.all_bits
            .select_ones(target_rank)
            .filter(|&idx| idx < self.len())
    }

    fn select_zeros(&self, target_rank: u64) -> Option<u64> {
        self.all_bits
            .select_zeros(target_rank)
            .filter(|&idx| idx < self.len())
    }
}

impl<'a> BitsMut for LeadingBitsOf<&'a mut [Word]> {
    fn replace(&mut self, idx: u64, with: bool) -> Option<bool> {
        if idx >= self.borrow().len() {
            return None;
        }
        let r = self.all_bits.replace(idx, with);
        debug_assert!(r.is_some(), "Should not be out of bounds");
        r
    }

    fn set(&mut self, idx: u64, to: bool) {
        if idx >= self.borrow().len() {
            panic!("Index is out of bounds for bits");
        }
        self.all_bits.set(idx, to);
    }
}

#[cfg(any(test, feature = "std", feature = "alloc"))]
impl BitsVec for LeadingBitsOf<Vec<Word>> {
    fn push(&mut self, bit: bool) {
        if self.skip_trailing_bits == 0 {
            assert_eq!(self.skipped_trailing_bits_count_ones, 0);
            self.all_bits
                .push(if bit { Word::msb() } else { Word::zeros() });
            self.skip_trailing_bits = Word::len() as u8 - 1;
        } else {
            let current_len = self.borrow().len();
            let prev_bit = self
                .all_bits
                .replace(current_len, bit)
                .expect("This index should not be out of bounds");
            self.skip_trailing_bits -= 1;
            self.skipped_trailing_bits_count_ones -= prev_bit as u8;
        }
    }
}

/// Bits stored as a sequence of words (most significant bit first).
#[derive(Debug, Copy, Clone)]
pub struct BitsOf<T> {
    leading_bits: LeadingBitsOf<T>,
    skip_leading_bits: u8,
    skipped_leading_bits_count_ones: u8,
}

pub type BitsRef<'a> = BitsOf<&'a [Word]>;
pub type BitsRefMut<'a> = BitsOf<&'a mut [Word]>;

const fn bits_from_leading<T>(leading_bits: LeadingBitsOf<T>) -> BitsOf<T> {
    BitsOf {
        leading_bits,
        skip_leading_bits: 0,
        skipped_leading_bits_count_ones: 0,
    }
}

impl<T> From<LeadingBitsOf<T>> for BitsOf<T> {
    fn from(leading_bits: LeadingBitsOf<T>) -> Self {
        bits_from_leading(leading_bits)
    }
}

impl<T> From<T> for BitsOf<T> {
    fn from(all_bits: T) -> Self {
        Self::from(<LeadingBitsOf<T>>::from(all_bits))
    }
}

impl<'a> BitsRef<'a> {
    pub const fn empty() -> Self {
        bits_from_leading(LeadingBitsOf::empty())
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    unsafe fn into_leading_bits_unchecked(self) -> LeadingBitsOf<&'a [Word]> {
        self.leading_bits
    }

    fn into_leading_bits(self) -> Option<LeadingBitsOf<&'a [Word]>> {
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

impl<'a> Bits for BitsRef<'a> {
    #[inline]
    fn len(&self) -> u64 {
        sub_should_not_overflow(self.leading_bits.len(), self.skip_leading_bits as u64)
    }

    fn get(&self, idx: u64) -> Option<bool> {
        // If this overflows then it must be out of range
        let actual_idx = idx.checked_add(self.skip_leading_bits as u64)?;
        self.leading_bits.get(actual_idx)
    }

    fn count_ones(&self) -> u64 {
        sub_should_not_overflow(
            self.leading_bits.count_ones(),
            self.skipped_leading_bits_count_ones as u64,
        )
    }

    fn rank_ones(&self, idx: u64) -> Option<u64> {
        // If this overflows then it must be out of range
        let actual_idx = idx.checked_add(self.skip_leading_bits as u64)?;
        self.leading_bits
            .rank_ones(actual_idx)
            .map(|base_rank_ones| {
                sub_should_not_overflow(base_rank_ones, self.skipped_leading_bits_count_ones as u64)
            })
    }

    fn select_ones(&self, target_rank: u64) -> Option<u64> {
        let skip_leading_bits = self.skip_leading_bits as u64;
        // If this overflows then we must be out of range
        let actual_target_rank =
            target_rank.checked_add(self.skipped_leading_bits_count_ones as u64)?;
        self.leading_bits
            .select_ones(actual_target_rank)
            .map(|base_select| sub_should_not_overflow(base_select, skip_leading_bits))
    }

    fn select_zeros(&self, target_rank: u64) -> Option<u64> {
        let skip_leading_bits = self.skip_leading_bits as u64;
        // If this overflows then we must be out of range
        let actual_target_rank = target_rank.checked_add(sub_should_not_overflow(
            skip_leading_bits,
            self.skipped_leading_bits_count_ones as u64,
        ))?;
        self.leading_bits
            .select_zeros(actual_target_rank)
            .map(|base_select| sub_should_not_overflow(base_select, skip_leading_bits))
    }
}

impl<'a> BitsMut for BitsRefMut<'a> {
    fn replace(&mut self, idx: u64, with: bool) -> Option<bool> {
        // If this overflows then it must be out of range
        let actual_idx = idx.checked_add(self.skip_leading_bits as u64)?;
        self.leading_bits.replace(actual_idx, with)
    }

    fn set(&mut self, idx: u64, to: bool) {
        // If this overflows then it must be out of range
        let actual_idx = match idx.checked_add(self.skip_leading_bits as u64) {
            Some(idx) => idx,
            None => panic!("Index is out of bounds for bits"),
        };
        self.leading_bits.set(actual_idx, to);
    }
}

impl<'a> BitsSplit for BitsRef<'a> {
    fn split_at(self, idx_bits: u64) -> Option<(Self, Self)> {
        fn split_leading_bits(
            leading_bits: LeadingBitsOf<&[Word]>,
            idx_bits: u64,
        ) -> Option<(LeadingBitsOf<&[Word]>, BitsRef)> {
            if idx_bits >= bits_len(&leading_bits) {
                return if idx_bits == bits_len(&leading_bits) {
                    Some((leading_bits, BitsRef::empty()))
                } else {
                    None
                }
            }
            let all_bits = leading_bits.all_bits;
            let (whole_words, bits) = crate::word::split_idx(idx_bits);
            if bits == 0 {
                let (first_part, second_part) = all_bits.split_at(whole_words);
                let second_part = LeadingBitsOf {
                    all_bits: second_part,
                    ..leading_bits
                };
                Some((first_part.into(), second_part.into()))
            } else {
                let first_part: &[Word] = all_bits
                    .get(..whole_words + 1)
                    .expect("This should not be out of bounds");
                let second_part: &[Word] = all_bits
                    .get(whole_words..)
                    .expect("This should not be out of bounds");
                let overlap_word: Word = *all_bits
                    .get(whole_words)
                    .expect("This should not be out of bounds");
                let overlap_full_count = overlap_word.count_ones();
                let overlap_first_part_count = overlap_word
                    .rank_ones(bits)
                    .expect("This should not be out of bounds");
                let overlap_second_part_count =
                    sub_should_not_overflow(overlap_full_count, overlap_first_part_count);
                let first_part = LeadingBitsOf {
                    all_bits: first_part,
                    skip_trailing_bits: (Word::len() - bits) as u8,
                    skipped_trailing_bits_count_ones: overlap_second_part_count as u8,
                };
                let second_part_leading = LeadingBitsOf {
                    all_bits: second_part,
                    ..leading_bits
                };
                let second_part = BitsOf {
                    leading_bits: second_part_leading,
                    skip_leading_bits: bits as u8,
                    skipped_leading_bits_count_ones: overlap_first_part_count as u8,
                };
                Some((first_part, second_part))
            }
        }

        // If this overflows then it must be out of range
        let actual_idx = idx_bits.checked_add(self.skip_leading_bits as u64)?;
        let BitsOf {
            leading_bits,
            skip_leading_bits,
            skipped_leading_bits_count_ones,
        } = self;
        split_leading_bits(leading_bits, actual_idx).map(|(leading_part, trailing_part)| {
            let leading_part = BitsOf {
                leading_bits: leading_part,
                skip_leading_bits,
                skipped_leading_bits_count_ones,
            };
            (leading_part, trailing_part)
        })
    }
}

#[cfg(any(test, feature = "std", feature = "alloc"))]
impl BitsVec for BitsOf<Vec<Word>> {
    fn push(&mut self, bit: bool) {
        self.leading_bits.push(bit)
    }
}

fn generalised_eq(left: BitsOf<&[Word]>, right: BitsOf<&[Word]>) -> bool {
    fn eq_in_range(
        left: &[Word],
        right: &[Word],
        left_offset: u64,
        right_offset: u64,
        length: u64,
    ) -> bool {
        let left_end_in_range = bits_len(left)
            .checked_sub(left_offset)
            .map_or(false, |available_length| available_length >= length);
        let right_end_in_range = bits_len(right)
            .checked_sub(right_offset)
            .map_or(false, |available_length| available_length >= length);

        if !(left_end_in_range & right_end_in_range) {
            panic!("Indexes out of bounds")
        }

        (0..length).into_iter().all(|idx| {
            let left_idx_bits = add_should_not_overflow(idx, left_offset);
            let right_idx_bits = add_should_not_overflow(idx, right_offset);
            let in_left = unsafe { crate::word::words_get_unchecked(left, left_idx_bits) };
            let in_right = unsafe { crate::word::words_get_unchecked(right, right_idx_bits) };
            in_left == in_right
        })
    }

    let left_len = left.len();
    if left_len != right.len() {
        return false;
    }
    eq_in_range(
        left.leading_bits.all_bits,
        right.leading_bits.all_bits,
        left.skip_leading_bits as u64,
        right.skip_leading_bits as u64,
        left_len,
    )
}

fn generalised_cmp(left: BitsOf<&[Word]>, right: BitsOf<&[Word]>) -> Ordering {
    fn cmp_in_range(
        left: &[Word],
        right: &[Word],
        left_offset: u64,
        right_offset: u64,
        length: u64,
    ) -> Ordering {
        let left_end_in_range = bits_len(left)
            .checked_sub(left_offset)
            .map_or(false, |available_length| available_length >= length);
        let right_end_in_range = bits_len(right)
            .checked_sub(right_offset)
            .map_or(false, |available_length| available_length >= length);

        if !(left_end_in_range & right_end_in_range) {
            panic!("Indexes out of bounds")
        }

        for idx in 0..length {
            let left_idx_bits = add_should_not_overflow(idx, left_offset);
            let right_idx_bits = add_should_not_overflow(idx, right_offset);
            let in_left = unsafe { crate::word::words_get_unchecked(left, left_idx_bits) };
            let in_right = unsafe { crate::word::words_get_unchecked(right, right_idx_bits) };
            let item_ordering = in_left.cmp(&in_right);
            match item_ordering {
                Ordering::Equal => continue,
                Ordering::Greater | Ordering::Less => return item_ordering,
            }
        }

        Ordering::Equal
    }

    cmp_in_range(
        left.leading_bits.all_bits,
        right.leading_bits.all_bits,
        left.skip_leading_bits as u64,
        right.skip_leading_bits as u64,
        min(left.len(), right.len()),
    )
    .then_with(|| left.len().cmp(&right.len()))
}

fn copy<T: Copy>(x: &T) -> T {
    *x
}

impl<'a> PartialEq for LeadingBitsOf<&'a [Word]> {
    fn eq(&self, other: &Self) -> bool {
        fn split_trailing_partial_word(
            mut bits: LeadingBitsOf<&[Word]>,
        ) -> (&[Word], LeadingBitsOf<&[Word]>) {
            if bits.skip_trailing_bits == 0 {
                (bits.all_bits, LeadingBitsOf::empty())
            } else {
                assert!(
                    bits.all_bits.len() > 0,
                    "If there are trailing bits they have to be in a word"
                );
                let (whole_words, trailing_word) = bits.all_bits.split_at(bits.all_bits.len() - 1);
                bits.all_bits = trailing_word;
                assert!(bits_len(bits.all_bits) > bits.skip_trailing_bits as u64);
                (whole_words, bits)
            }
        }

        other.skip_trailing_bits == self.skip_trailing_bits
            && self.all_bits.len() == other.all_bits.len()
            && {
                debug_assert_eq!(self.len(), other.len());
                let (self_whole, self_partial) = split_trailing_partial_word(copy(self));
                let (other_whole, other_partial) = split_trailing_partial_word(copy(other));
                self_whole == other_whole
                    && generalised_eq(self_partial.into(), other_partial.into())
            }
    }
}

impl<'a> Eq for LeadingBitsOf<&'a [Word]> {}

impl<'a> PartialOrd for LeadingBitsOf<&'a [Word]> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for LeadingBitsOf<&'a [Word]> {
    fn cmp(&self, other: &Self) -> Ordering {
        fn split_trailing_partial_words(
            mut bits: LeadingBitsOf<&[Word]>,
            n: usize,
        ) -> (&[Word], LeadingBitsOf<&[Word]>) {
            let (whole_words, final_words) = bits.all_bits.split_at(n);
            bits.all_bits = final_words;
            assert!(bits_len(bits.all_bits) >= bits.skip_trailing_bits as u64);
            (whole_words, bits)
        }

        let initial_whole_words = (min(self.len(), other.len()) / Word::len()) as usize;
        let (self_whole, self_partial) =
            split_trailing_partial_words(copy(self), initial_whole_words);
        let (other_whole, other_partial) =
            split_trailing_partial_words(copy(other), initial_whole_words);
        debug_assert_eq!(self_whole.len(), other_whole.len());
        <[Word] as Ord>::cmp(self_whole, other_whole)
            .then_with(|| generalised_cmp(self_partial.into(), other_partial.into()))
    }
}

fn split_leading_partial_word(bits: BitsRef) -> (BitsRef, LeadingBitsOf<&[Word]>) {
    let first_whole_word_index =
        sub_should_not_overflow(Word::len(), bits.skip_leading_bits as u64) % Word::len();
    match bits.split_at(first_whole_word_index) {
        None => (bits, LeadingBitsOf::empty()),
        Some((partial, whole_words)) => {
            debug_assert!(whole_words.into_leading_bits().is_some());
            let whole_words = unsafe { whole_words.into_leading_bits_unchecked() };
            (partial, whole_words)
        }
    }
}

impl<'a> PartialEq for BitsRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        if self.skip_leading_bits == other.skip_leading_bits {
            let (self_partial, self_whole_words) = split_leading_partial_word(copy(self));
            let (other_partial, other_whole_words) = split_leading_partial_word(copy(other));
            generalised_eq(self_partial, other_partial) && self_whole_words == other_whole_words
        } else {
            generalised_eq(copy(self), copy(other))
        }
    }
}

impl<'a> Eq for BitsRef<'a> {}

impl<'a> PartialOrd for BitsRef<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a> Ord for BitsRef<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.skip_leading_bits == other.skip_leading_bits {
            let (self_partial, self_whole_words) = split_leading_partial_word(copy(self));
            let (other_partial, other_whole_words) = split_leading_partial_word(copy(other));
            generalised_cmp(self_partial, other_partial)
                .then_with(|| self_whole_words.cmp(&other_whole_words))
        } else {
            generalised_cmp(copy(self), copy(other))
        }
    }
}

impl<'a> PartialEq for BitsRefMut<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().eq(&other.borrow())
    }

    fn ne(&self, other: &Self) -> bool {
        self.borrow().ne(&other.borrow())
    }
}

impl<'a> Eq for BitsRefMut<'a> {}

impl<'a> PartialOrd for BitsRefMut<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.borrow().partial_cmp(&other.borrow())
    }

    fn lt(&self, other: &Self) -> bool {
        self.borrow().lt(&other.borrow())
    }

    fn le(&self, other: &Self) -> bool {
        self.borrow().le(&other.borrow())
    }

    fn gt(&self, other: &Self) -> bool {
        self.borrow().gt(&other.borrow())
    }

    fn ge(&self, other: &Self) -> bool {
        self.borrow().ge(&other.borrow())
    }
}

impl<'a> Ord for BitsRefMut<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.borrow().cmp(&other.borrow())
    }
}

#[cfg(any(test, feature = "std", feature = "alloc"))]
impl PartialEq for BitsOf<Vec<Word>> {
    fn eq(&self, other: &Self) -> bool {
        self.borrow().eq(&other.borrow())
    }

    fn ne(&self, other: &Self) -> bool {
        self.borrow().eq(&other.borrow())
    }
}

#[cfg(any(test, feature = "std", feature = "alloc"))]
impl Eq for BitsOf<Vec<Word>> {}

#[cfg(any(test, feature = "std", feature = "alloc"))]
impl PartialOrd for BitsOf<Vec<Word>> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.borrow().partial_cmp(&other.borrow())
    }

    fn lt(&self, other: &Self) -> bool {
        self.borrow().lt(&other.borrow())
    }

    fn le(&self, other: &Self) -> bool {
        self.borrow().le(&other.borrow())
    }

    fn gt(&self, other: &Self) -> bool {
        self.borrow().gt(&other.borrow())
    }

    fn ge(&self, other: &Self) -> bool {
        self.borrow().ge(&other.borrow())
    }
}

#[cfg(any(test, feature = "std", feature = "alloc"))]
impl Ord for BitsOf<Vec<Word>> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.borrow().cmp(&other.borrow())
    }
}

impl<T: crate::import::Borrow<[Word]>> LeadingBitsOf<T> {
    fn borrow(&self) -> LeadingBitsOf<&[Word]> {
        let LeadingBitsOf {
            all_bits,
            skip_trailing_bits,
            skipped_trailing_bits_count_ones,
        } = self;
        LeadingBitsOf {
            all_bits: all_bits.borrow(),
            skip_trailing_bits: *skip_trailing_bits,
            skipped_trailing_bits_count_ones: *skipped_trailing_bits_count_ones,
        }
    }
}

impl<T: crate::import::Borrow<[Word]>> BitsOf<T> {
    pub fn borrow(&self) -> BitsRef {
        let BitsOf {
            leading_bits,
            skip_leading_bits,
            skipped_leading_bits_count_ones,
        } = self;
        BitsOf {
            leading_bits: leading_bits.borrow(),
            skip_leading_bits: *skip_leading_bits,
            skipped_leading_bits_count_ones: *skipped_leading_bits_count_ones,
        }
    }
}

impl<T: crate::import::BorrowMut<[Word]>> LeadingBitsOf<T> {
    fn borrow_mut(&mut self) -> LeadingBitsOf<&mut [Word]> {
        let LeadingBitsOf {
            all_bits,
            skip_trailing_bits,
            skipped_trailing_bits_count_ones,
        } = self;
        LeadingBitsOf {
            all_bits: all_bits.borrow_mut(),
            skip_trailing_bits: *skip_trailing_bits,
            skipped_trailing_bits_count_ones: *skipped_trailing_bits_count_ones,
        }
    }
}

impl<T: crate::import::BorrowMut<[Word]>> BitsOf<T> {
    pub fn borrow_mut(&mut self) -> BitsRefMut {
        let BitsOf {
            leading_bits,
            skip_leading_bits,
            skipped_leading_bits_count_ones,
        } = self;
        BitsOf {
            leading_bits: leading_bits.borrow_mut(),
            skip_leading_bits: *skip_leading_bits,
            skipped_leading_bits_count_ones: *skipped_leading_bits_count_ones,
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

impl<'a> Iterator for ChunksIter<'a> {
    type Item = BitsRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.data.split_at(self.bits_in_chunk) {
            None => {
                if self.data.is_empty() {
                    None
                } else {
                    Some(replace(&mut self.data, BitsOf::empty()))
                }
            }
            Some(split) => {
                self.data = split.1;
                Some(split.0)
            }
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

impl<'a> crate::import::iter::ExactSizeIterator for ChunksIter<'a> {
    fn len(&self) -> usize {
        ceil_div_u64(self.data.len(), self.bits_in_chunk) as usize
    }
}

impl<'a> Bits for BitsRefMut<'a> {
    fn len(&self) -> u64 {
        self.borrow().len()
    }

    fn get(&self, idx: u64) -> Option<bool> {
        self.borrow().get(idx)
    }

    fn count_ones(&self) -> u64 {
        self.borrow().count_ones()
    }

    fn count_zeros(&self) -> u64 {
        self.borrow().count_zeros()
    }

    fn rank_ones(&self, idx: u64) -> Option<u64> {
        self.borrow().rank_ones(idx)
    }

    fn rank_zeros(&self, idx: u64) -> Option<u64> {
        self.borrow().rank_zeros(idx)
    }

    fn select_ones(&self, target_rank: u64) -> Option<u64> {
        self.borrow().select_ones(target_rank)
    }

    fn select_zeros(&self, target_rank: u64) -> Option<u64> {
        self.borrow().select_zeros(target_rank)
    }
}

#[cfg(any(test, feature = "std", feature = "alloc"))]
impl Bits for BitsOf<Vec<Word>> {
    fn len(&self) -> u64 {
        self.borrow().len()
    }

    fn get(&self, idx: u64) -> Option<bool> {
        self.borrow().get(idx)
    }

    fn count_ones(&self) -> u64 {
        self.borrow().count_ones()
    }

    fn count_zeros(&self) -> u64 {
        self.borrow().count_zeros()
    }

    fn rank_ones(&self, idx: u64) -> Option<u64> {
        self.borrow().rank_ones(idx)
    }

    fn rank_zeros(&self, idx: u64) -> Option<u64> {
        self.borrow().rank_zeros(idx)
    }

    fn select_ones(&self, target_rank: u64) -> Option<u64> {
        self.borrow().select_ones(target_rank)
    }

    fn select_zeros(&self, target_rank: u64) -> Option<u64> {
        self.borrow().select_zeros(target_rank)
    }
}

#[cfg(any(test, feature = "std", feature = "alloc"))]
impl BitsMut for BitsOf<Vec<Word>> {
    fn replace(&mut self, idx: u64, with: bool) -> Option<bool> {
        self.borrow_mut().replace(idx, with)
    }

    fn set(&mut self, idx: u64, to: bool) {
        self.borrow_mut().set(idx, to)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::bits_traits::tests as bits_tests;
    use crate::word::tests::words;
    use proptest::collection::SizeRange;
    use proptest::prelude::*;

    fn get_bit<T: Bits + ?Sized>(t: &T, idx: u64) -> Option<bool> {
        t.get(idx)
    }

    pub fn bits(n_words: impl Into<SizeRange>) -> impl Strategy<Value = BitsOf<Vec<Word>>> {
        words(n_words).prop_flat_map(|all_bits| {
            let ranges = {
                let word_len = Word::len() as u8;
                let range = if all_bits.len() == 1 {
                    0..=word_len
                } else {
                    0..=word_len - 1
                };
                (range.clone(), range)
            };
            (Just(all_bits), ranges).prop_map(|(all_bits, (mut a, mut b))| {
                if all_bits.is_empty() {
                    BitsOf::from(all_bits)
                } else {
                    let (skip_leading_bits, skip_trailing_bits) = if all_bits.len() <= 1 {
                        if b < a {
                            swap(&mut a, &mut b)
                        }
                        (a, b - a)
                    } else {
                        (a, b)
                    };
                    let first_word = *all_bits.first().expect("Is not empty");
                    let last_word = *all_bits.last().expect("Is not empty");
                    let leading_bits = LeadingBitsOf {
                        all_bits,
                        skip_trailing_bits,
                        skipped_trailing_bits_count_ones: if skip_trailing_bits > 0 {
                            (last_word.count_ones()
                                - last_word
                                    .rank_ones(Word::len() - skip_trailing_bits as u64)
                                    .expect("Should not be out of bounds"))
                                as u8
                        } else {
                            0
                        },
                    };
                    BitsOf {
                        leading_bits,
                        skip_leading_bits,
                        skipped_leading_bits_count_ones: {
                            (first_word
                                .rank_ones(skip_leading_bits as u64)
                                .expect("Should not be out of bounds"))
                                as u8
                        },
                    }
                }
            })
        })
    }

    fn some_bits() -> impl Strategy<Value = BitsOf<Vec<Word>>> {
        bits(0..=64)
    }

    proptest! {
        #[test]
        fn bits_bits_test_len(bits in bits(0..=1024)) {
            let bits = bits.borrow();
            prop_assert!(bits.leading_bits.len() <= bits_len(bits.leading_bits.all_bits));
            prop_assert!(bits.len() <= bits.leading_bits.len());
            prop_assert_eq!(bits.leading_bits.len() + bits.leading_bits.skip_trailing_bits as u64, bits_len(bits.leading_bits.all_bits));
            prop_assert_eq!(bits.len() + bits.skip_leading_bits as u64, bits.leading_bits.len());
        }

        #[test]
        fn bits_bits_test_get(bits in some_bits()) {
            let bits = bits.borrow();
            let leading = bits.leading_bits;
            for i in 0..leading.len() {
                prop_assert!(leading.get(i).is_some());
                prop_assert_eq!(leading.get(i), get_bit(leading.all_bits, i));
            }
            assert!(leading.get(leading.len()).is_none());

            for i in 0..bits.len() {
                prop_assert!(bits.get(i).is_some());
                prop_assert_eq!(bits.get(i), leading.get(i + bits.skip_leading_bits as u64));
            }
            prop_assert!(bits.get(bits.len()).is_none());
        }

        #[test]
        fn bits_bits_test_count(bits in some_bits()) {
            let bits = bits.borrow();
            bits_tests::from_get_and_len::test_count(&bits);
        }

        #[test]
        fn bits_bits_test_rank(bits in some_bits()) {
            let bits = bits.borrow();
            bits_tests::from_get_and_len::test_rank(&bits);
        }

        #[test]
        fn bits_bits_test_select(bits in some_bits()) {
            let bits = bits.borrow();
            bits_tests::from_get_and_len::test_select(&bits);
        }

        #[test]
        fn bits_bits_mut_test_replace(mut bits in some_bits()) {
            let mut bits = bits.borrow_mut();
            bits_tests::from_get_and_len::test_replace(&mut bits);
        }

        #[test]
        fn bits_bits_mut_test_set(mut bits in some_bits()) {
            let mut bits = bits.borrow_mut();
            bits_tests::from_get_and_len::test_set_in_bounds(&mut bits);
        }

        #[test]
        fn bits_bits_split_test_split_absent(bits in some_bits()) {
            let bits: BitsRef = bits.borrow();
            prop_assert!(bits.split_at(bits.len() + 1).is_none());
        }

        #[test]
        fn bits_bits_split_test_split_present((bits, split_at) in some_bits().prop_flat_map(|bits| { let len = bits.len(); (Just(bits), 0..=len)})) {
            let bits: BitsRef = bits.borrow();
            let split = bits.split_at(split_at);
            prop_assert!(split.is_some());
            let (left, right) = split.unwrap();
            prop_assert_eq!(split_at, left.len());
            prop_assert_eq!(bits.len(), left.len() + right.len());
            for i in 0..left.len() {
                prop_assert_eq!(bits.get(i), left.get(i));
            }
            for i in 0..right.len() {
                prop_assert_eq!(bits.get(split_at + i), right.get(i));
            }
        }

        #[test]
        fn bits_bits_vec_test_push(mut bits in some_bits()) {
            bits_tests::from_get_and_len::test_push(&mut bits);
        }
    }

    #[test]
    fn empty_is_empty() {
        let empty = BitsRef::empty();
        assert_eq!(0, empty.len());
        assert_eq!(None, empty.get(0));
        assert_eq!(true, empty.is_empty());
    }

    fn check_chunks(bits: BitsRef, chunk_size: u64) {
        let mut offset = 0;
        let mut saw_last_chunk = false;
        let mut saw_count = 0;
        for chunk in bits
            .chunks(chunk_size)
            .expect("Chunk size should not be zero")
        {
            assert!(!saw_last_chunk);
            let chunk_len = bits_len(&chunk);
            if chunk_len < chunk_size {
                saw_last_chunk = true;
            }
            for idx in 0..chunk_len {
                assert!(chunk.get(idx).is_some());
                assert_eq!(chunk.get(idx), bits.get(idx + offset));
            }
            assert!(chunk.get(chunk_len).is_none());
            offset += chunk_len;
            saw_count += 1;
        }
        assert_eq!(offset, bits.len());
        assert_eq!(bits.chunks(chunk_size).unwrap().len(), saw_count);
    }

    proptest! {
        #[test]
        fn bits_chunks(bits in some_bits(), chunk_size in 1..=999999999u64) {
            check_chunks(bits.borrow(), chunk_size);
        }
    }

    fn to_bool_vec<T: Bits + ?Sized>(bits: &T) -> Vec<bool> {
        (0..bits.len()).map(|idx| bits.get(idx).expect("Should not be out of bounds")).collect()
    }

    fn eq_by_vec<L: Bits + ?Sized, R: Bits + ?Sized>(l: &L, r: &R) -> bool {
        <Vec<bool> as PartialEq>::eq(&to_bool_vec(l), &to_bool_vec(r))
    }

    fn cmp_by_vec<L: Bits + ?Sized, R: Bits + ?Sized>(l: &L, r: &R) -> Ordering {
        <Vec<bool> as Ord>::cmp(&to_bool_vec(l), &to_bool_vec(r))
    }


    proptest! {
        #[test]
        fn bits_eq_test_likely_ne(l in some_bits(), r in some_bits()) {
            prop_assert_eq!(l == r, eq_by_vec(&l, &r));
        }

        #[test]
        fn bits_eq_test_eq(x in some_bits()) {
            prop_assert_eq!(x == x, eq_by_vec(&x, &x));
        }

        #[test]
        fn bits_cmp_test_likely_ne(l in some_bits(), r in some_bits()) {
            prop_assert_eq!(l.cmp(&r), cmp_by_vec(&l, &r));
        }

        #[test]
        fn bits_cmp_test_eq(x in some_bits()) {
            prop_assert_eq!(x.cmp(&x), cmp_by_vec(&x, &x));
        }
    }
}
