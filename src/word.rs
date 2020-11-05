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
//! Tools for working with a single (64bit) word as bits.
use crate::import::prelude::*;

#[inline]
fn slice_len<T>(slice: &[T]) -> usize {
    slice.len()
}

#[inline]
fn bits_len<T: Bits + ?Sized>(bits: &T) -> u64 { bits.len() }

const fn bits_in_size<T: Sized>() -> u64 {
    size_of::<T>() as u64 * 8
}

pub(crate) type WordStorage = u32;

/// The bits of a single word, as a sequence from MSB to LSB.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Word(WordStorage);

impl From<WordStorage> for Word {
    fn from(word: u32) -> Self {
        Word(word)
    }
}

impl From<Word> for WordStorage {
    fn from(word: Word) -> Self {
        word.0
    }
}

impl Word {
    pub const ZEROS: Self = Self(0);
    pub const fn zeros() -> Self { Self::ZEROS }

    pub const ONES: Self = Self(!0);
    pub const fn ones() -> Self { Self::ONES }

    pub const fn len() -> u64 {
        bits_in_size::<WordStorage>()
    }

    pub(crate) const fn msb() -> Self {
        Self(1 << (Self::len() as WordStorage - 1))
    }

    pub const fn flip_bits(self) -> Self {
        Self(!self.0)
    }

    fn get_direct(self, idx: u64) -> Option<bool> {
        if idx >= bits_len(&self) {
            None
        } else {
            Some(((self.0 << idx as WordStorage) & Self::msb().0) != 0)
        }
    }

    fn rank_ones_direct(self, idx: u64) -> Option<u64> {
        let len = bits_len(&self);
        if idx >= len {
            None
        } else if idx == 0 {
            Some(0)
        } else {
            Some((self.0 >> (len - idx) as WordStorage).count_ones() as u64)
        }
    }

    fn select_ones_direct(self, target_rank: u64) -> Option<u64> {
        let len = bits_len(&self) as u32;
        let mut target_rank = if target_rank >= len as u64 {
            return None;
        } else {
            target_rank as u32
        };
        let mut from = self.0;
        let mut offset = 0u32;

        loop {
            let leading_zeros = from.leading_zeros();
            if leading_zeros >= len {
                return None;
            } else if target_rank == 0 {
                return Some((offset + leading_zeros) as u64)
            }

            from <<= leading_zeros;
            offset += leading_zeros;

            let part_count =
                Self(from).rank_ones_direct(target_rank as u64)
                    .expect("If this is out of range it's a bug") as u32;
            from <<= target_rank;
            offset += target_rank;
            target_rank -= part_count;
        }
    }

    pub fn zero_after_first_bits(self, n: u64) -> Option<Self> {
        if n > bits_len(&self) {
            None
        } else {
            let mask = !(Self::ONES.0 >> n);
            Some(Self(self.0 & mask))
        }
    }
}

impl Bits for Word {
    #[inline]
    fn len(&self) -> u64 {
        Self::len()
    }

    fn get(&self, idx: u64) -> Option<bool> {
        self.get_direct(idx)
    }

    fn count_ones(&self) -> u64 {
        self.0.count_ones() as u64
    }

    fn count_zeros(&self) -> u64 {
        self.0.count_zeros() as u64
    }

    fn rank_ones(&self, idx: u64) -> Option<u64> {
        self.rank_ones_direct(idx)
    }

    fn rank_zeros(&self, idx: u64) -> Option<u64> {
        self.flip_bits().rank_ones_direct(idx)
    }

    fn select_ones(&self, target_rank: u64) -> Option<u64> {
        self.select_ones_direct(target_rank)
    }

    fn select_zeros(&self, target_rank: u64) -> Option<u64> {
        self.flip_bits().select_ones_direct(target_rank)
    }
}

impl BitsMut for Word {
    fn replace(&mut self, idx: u64, with: bool) -> Option<bool> {
        if idx >= bits_len(self) {
            return None;
        }
        let original_storage = self.0;
        let mask = Self::msb().0 >> idx;
        let other_bits = original_storage & !mask;
        let new_bit = if with { mask } else { 0 };
        self.0 = new_bit | other_bits;
        Some((original_storage & mask) != 0)
    }
}

impl crate::import::Default for Word {
    fn default() -> Self {
        Self::zeros()
    }
}


pub(crate) fn split_idx(idx_bits: u64) -> (usize, u64) {
    ((idx_bits / Word::len()) as usize, (idx_bits % Word::len()))
}

pub(crate) unsafe fn words_get_unchecked(words: &[Word], idx: u64) -> bool {
    let (word_idx, bit_idx) = split_idx(idx);
    let word = {
        #[cfg(test)] {
            debug_assert!(words.get(word_idx).is_some());
        };
        words.get_unchecked(word_idx)
    };
    word.get(bit_idx).expect("The bit index should not be out of bounds")
}

impl Bits for [Word] {
    fn len(&self) -> u64 {
        slice_len(self) as u64 * Word::len()
    }

    fn get(&self, idx: u64) -> Option<bool> {
        if idx >= bits_len(self) {
            None
        } else {
            Some(unsafe { words_get_unchecked(self, idx) })
        }
    }

    fn count_ones(&self) -> u64 {
        self.iter().map(|word| word.count_ones() as u64).sum()
    }

    fn rank_ones(&self, idx: u64) -> Option<u64> {
        let (word_idx, bit_idx) = split_idx(idx);
        if word_idx >= slice_len(self) { return None; }
        let (words_before, words_from) = self.split_at(word_idx);
        let words_rank = words_before.count_ones();
        let last_word_rank = {
            debug_assert!(words_from.get(0).is_some(), "This should be in bounds - we already checked");
            let last_word = unsafe { words_from.get_unchecked(0) };
            last_word.rank_ones(bit_idx).expect("This should be in bounds")
        };
        Some(words_rank + last_word_rank)
    }

    fn select_ones(&self, mut target_rank: u64) -> Option<u64> {
        let mut words_iter = self.iter();
        let mut offset = 0u64;
        loop {
            let &word = words_iter.next()?;
            let count = word.count_ones();
            if count > target_rank {
                let within_word =
                    word.select_ones(target_rank)
                        .expect("This should be in bounds - we already checked");
                return Some(offset + within_word)
            } else {
                target_rank -= count;
                offset += word.len();
            }
        }
    }

    fn select_zeros(&self, mut target_rank: u64) -> Option<u64> {
        let mut words_iter = self.iter();
        let mut offset = 0u64;
        loop {
            let &word = words_iter.next()?;
            let count = word.count_zeros();
            if count > target_rank {
                let within_word =
                    word.select_zeros(target_rank)
                        .expect("This should be in bounds - we already checked");
                return Some(offset + within_word)
            } else {
                target_rank -= count;
                offset += word.len();
            }
        }
    }
}

impl BitsMut for [Word] {
    fn replace(&mut self, idx: u64, with: bool) -> Option<bool> {
        let (word_idx, bit_idx) = split_idx(idx);
        let word = self.get_mut(word_idx)?;
        let r = word.replace(bit_idx, with);
        debug_assert!(r.is_some(), "This should be in bounds");
        r
    }

    fn set(&mut self, idx: u64, to: bool) {
        let (word_idx, bit_idx) = split_idx(idx);
        let word = match self.get_mut(word_idx) {
            Some(word) => word,
            None => panic!("Index is out of bounds for bits"),
        };
        word.set(bit_idx, to);
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::bits_traits::tests as bits_tests;
    use proptest::prelude::*;

    fn get_bit<T: Bits + ?Sized>(t: &T, idx: u64) -> Option<bool> {
        t.get(idx)
    }

    impl Arbitrary for Word {
        type Parameters = ();

        fn arbitrary_with(args: Self::Parameters) -> Self::Strategy {
            let () = args;
            <WordStorage as Arbitrary>::arbitrary().prop_map_into()
        }

        type Strategy = proptest::strategy::MapInto<<WordStorage as Arbitrary>::Strategy, Word>;
    }

    proptest! {
        #[test]
        fn word_bits_len(word in any::<Word>()) {
            assert_eq!(word.len(), 32);
        }
    }

    #[test]
    fn word_bits_get() {
        for i in 0..Word::len() {
            assert_eq!(Word::zeros().get(i), Some(false));
            assert_eq!(Word::ones().get(i), Some(true));
        }

        assert!(Word::zeros().get(Word::len()).is_none());
        assert!(Word::ones().get(Word::len()).is_none());

        for idx in 0..Word::len() {
            let msb: WordStorage = 0b1000_0000_0000_0000_0000_0000_0000_0000;
            assert!(msb.checked_mul(2).is_none());
            assert_eq!(msb.count_ones(), 1);
            let example_word = Word(msb >> idx);
            for i in 0..Word::len() {
                assert_eq!(example_word.get(i), Some(i == idx));
            }
            assert!(example_word.get(Word::len()).is_none());
        }
    }

    proptest! {
        #[test]
        fn word_bits_count(word in any::<Word>()) {
            bits_tests::from_get_and_len::test_count(&word);
        }

        #[test]
        fn word_bits_rank(word in any::<Word>()) {
            bits_tests::from_get_and_len::test_rank(&word);
        }

        #[test]
        fn word_bits_select(word in any::<Word>()) {
            bits_tests::from_get_and_len::test_select(&word);
        }

        #[test]
        fn word_bits_mut_replace(mut word in any::<Word>()) {
            bits_tests::from_get_and_len::test_replace(&mut word);
        }

        #[test]
        fn word_bits_mut_test_set(mut word in any::<Word>()) {
            bits_tests::from_get_and_len::test_set_in_bounds(&mut word);
        }


    }

    pub fn words(size: impl Into<proptest::collection::SizeRange>) -> impl Strategy<Value = Vec<Word>> {
        proptest::collection::vec(any::<Word>(), size)
    }

    fn some_words() -> impl Strategy<Value = Vec<Word>> {
        words(0..=64)
    }

    proptest! {
        #[test]
        fn words_bits_len(words in some_words()) {
            assert_eq!(bits_len(words.as_slice()), slice_len(words.as_slice()) as u64 * 32);
        }
    }

    #[test]
    fn words_bits_get() {
        let ones = vec![Word::ONES; 16];
        let ones = ones.as_slice();
        (0..bits_len(ones)).for_each(|idx| {
            assert_eq!(get_bit(ones, idx), Some(true));
        });
        assert!(get_bit(ones, bits_len(ones)).is_none());

        let zeros = vec![Word::ZEROS; 16];
        let zeros = zeros.as_slice();
        (0..bits_len(zeros)).for_each(|idx| {
            assert_eq!(get_bit(zeros, idx), Some(false));
        });
        assert!(get_bit(zeros, bits_len(zeros)).is_none());

        let mut pattern = vec![Word::ZEROS; 3];
        pattern[0].set(4, true);
        pattern[1].set(5, true);
        pattern[2].set(6, true);
        let pattern = pattern.as_slice();
        assert_eq!(get_bit(pattern, 0 * Word::len() + 4), Some(true));
        assert_eq!(get_bit(pattern, 1 * Word::len() + 5), Some(true));
        assert_eq!(get_bit(pattern, 2 * Word::len() + 6), Some(true));
    }

    proptest! {
        #[test]
        fn words_bits_count(words in some_words()) {
            bits_tests::from_get_and_len::test_count(words.as_slice());
        }

        #[test]
        fn words_bits_rank(words in some_words()) {
            bits_tests::from_get_and_len::test_rank(words.as_slice());
        }

        #[test]
        fn words_bits_select(words in some_words()) {
            bits_tests::from_get_and_len::test_select(words.as_slice());
        }

        #[test]
        fn words_bits_mut_replace(mut words in some_words()) {
            bits_tests::from_get_and_len::test_replace(words.as_mut_slice());
        }

        #[test]
        fn words_bits_mut_test_set(mut words in some_words()) {
            bits_tests::from_get_and_len::test_set_in_bounds(words.as_mut_slice());
        }
    }

    fn expected_lexicographic_ordering(l: Word, r: Word) -> Ordering {
        for i in 0..Word::len() {
            let cmp = l.get(i).unwrap().cmp(&r.get(i).unwrap());
            match cmp {
                Ordering::Equal => continue,
                Ordering::Less | Ordering::Greater => return cmp,
            }
        }
        Ordering::Equal
    }

    proptest! {
        #[test]
        fn words_ord_is_lexicographic((l, r) in any::<(Word, Word)>()) {
            assert_eq!(l.cmp(&r), expected_lexicographic_ordering(l, r));
        }
    }
}



