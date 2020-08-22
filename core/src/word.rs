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
//! Tools for working with a single (64bit) word as bits.
use crate::ones_or_zeros::{OnesOrZeros, ZeroBits};

// TODO: Go through and refresh this file

/// The 64 bits of a single word, as a sequence from MSB to LSB.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Word(u64);

impl From<u64> for Word {
    fn from(i: u64) -> Self {
        Word(i)
    }
}

impl From<Word> for u64 {
    fn from(i: Word) -> Self {
        i.0
    }
}

impl Word {
    /// The word where all bits are zero.
    pub const ZEROS: Self = Word(0);

    /// The number of bits in one word (always 64).
    #[inline]
    pub fn len(self) -> usize {
        64
    }

    /// Check that an index is in bounds.
    #[inline]
    fn index_check(self, idx: usize) -> Option<()> {
        if idx >= self.len() {
            None
        } else {
            Some(())
        }
    }

    /// Get a single bit by index.
    ///
    /// Returns `None` if the index is out of bounds.
    #[inline]
    pub fn get(self, idx: usize) -> Option<bool> {
        self.index_check(idx)?;
        Some((u64::from(self) & (1u64 << (63 - idx) as u64)) > 0)
    }

    /// Set a single bit by creating a new word.
    ///
    /// Returns `None` if the index is out of bounds.
    pub fn set_copy(self, idx: usize, to: bool) -> Option<Self> {
        self.index_check(idx)?;
        let mask = 1u64 << (63 - idx) as u64;
        let int = u64::from(self);
        let res = if to { int | mask } else { int & (!mask) };
        Some(res.into())
    }

    /// Set a single bit in place.
    ///
    /// Returns `None` (and makes no change) if the index is out of bounds.
    #[inline]
    pub fn set(&mut self, idx: usize, to: bool) -> Option<()> {
        match self.set_copy(idx, to) {
            None => None,
            Some(rep) => Some(*self = rep),
        }
    }
}

pub(crate) fn select_ones_u16(from: u16, mut nth: u32) -> u32 {
    let mut offset = 0;
    let mut from = from as u32;
    loop {
        let n_leading_zeros = from.leading_zeros() - 16;
        if n_leading_zeros >= 16 {
            return 16;
        } else if nth == 0 {
            return n_leading_zeros + offset;
        };

        let shift = n_leading_zeros + nth;
        from <<= shift;
        offset += shift;
        nth -= (from & 0xffff0000).count_ones();
        from &= 0x0000ffff;
    }
}

impl Word {
    /// Invert all the bits in the word.
    #[inline]
    pub fn complement(self) -> Self {
        Self::from(!u64::from(self))
    }

    #[inline]
    fn desired_bits_as_ones<W: OnesOrZeros>(self) -> Self {
        if W::is_ones() {
            self
        } else {
            self.complement()
        }
    }

    /// Count the set bits.
    #[inline(always)]
    pub fn count_ones(self) -> u32 {
        u64::from(self).count_ones()
    }

    /// Count the unset bits.
    #[inline(always)]
    pub fn count_zeros(self) -> u32 {
        u64::from(self).count_zeros()
    }

    /// Count the set bits before a position in the word.
    ///
    /// Returns `None` if the index is out of bounds.
    #[inline]
    pub fn rank_ones(self, idx: usize) -> Option<u32> {
        if idx == 0 {
            return Some(0);
        };
        self.index_check(idx)?;
        let to_count = u64::from(self) >> (64 - idx) as u64;
        Some(to_count.count_ones())
    }

    /// Count the unset bits before a position in the word.
    ///
    /// Returns `None` if the index is out of bounds.
    #[inline]
    pub fn rank_zeros(self, idx: usize) -> Option<u32> {
        self.complement().rank_ones(idx)
    }

    /// Find the position of a set bit by its rank.
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank_ones(result) == Some(target_rank)`
    /// and `get(result) == Some(true)`.
    pub fn select_ones(self, nth: u32) -> Option<u32> {
        let rank_32 = self.rank_ones(32)?;
        let rank_16 = self.rank_ones(16)?;
        let rank_48 = self.rank_ones(48)?;
        let int: u64 = self.into();

        let res = if rank_32 > nth {
            if rank_16 > nth {
                select_ones_u16(((int >> 48) & 0xffff) as u16, nth)
            } else {
                select_ones_u16(((int >> 32) & 0xffff) as u16, nth - rank_16) + 16
            }
        } else {
            if rank_48 > nth {
                select_ones_u16(((int >> 16) & 0xffff) as u16, nth - rank_32) + 32
            } else {
                if nth >= self.count_ones() {
                    return None;
                }

                select_ones_u16((int & 0xffff) as u16, nth - rank_48) + 48
            }
        };
        Some(res)
    }

    #[inline]
    pub(crate) fn select<W: OnesOrZeros>(self, nth: u32) -> Option<u32> {
        self.desired_bits_as_ones::<W>().select_ones(nth)
    }

    /// Find the position of a set bit by its rank.
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank_zeros(result) == Some(target_rank)`
    /// and `get(result) == Some(false)`.
    #[inline]
    pub fn select_zeros(self, nth: u32) -> Option<u32> {
        self.select::<ZeroBits>(nth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::Arbitrary;
    use std::boxed::Box;
    use std::vec::Vec;

    impl Arbitrary for Word {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            u64::arbitrary(g).into()
        }

        fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
            let base = u64::from(*self).shrink();
            Box::new(base.map(|x| x.into()))
        }
    }

    #[test]
    fn test_get() {
        let get_bits = |x: u64| {
            let x = Word::from(x);
            let mut res = Vec::with_capacity(x.len());
            for i in 0..x.len() {
                res.push(x.get(i).unwrap());
            }
            res.into_iter().enumerate()
        };
        for (i, x) in get_bits(0xffffffffffffffff) {
            assert!(x, "Bit {} was wrong", i);
        }
        for (i, x) in get_bits(0x0000000000000000) {
            assert!(!x, "Bit {} was wrong", i);
        }
        for (i, x) in get_bits(0xffffffff00000000) {
            assert_eq!(x, i < 32, "Bit {} was wrong", i);
        }
        for (i, x) in get_bits(0x00000000ffffffff) {
            assert_eq!(x, i >= 32, "Bit {} was wrong", i);
        }
        for (i, x) in get_bits(0xaaaaaaaaaaaaaaaa) {
            assert_eq!(x, (i & 1) == 0, "Bit {} was wrong", i);
        }
        for (i, x) in get_bits(0x5555555555555555) {
            assert_eq!(x, (i & 1) > 0, "Bit {} was wrong", i);
        }

        for shift in 0..(Word::from(0).len() as u64) {
            for (i, x) in get_bits(0x8000000000000000u64 >> shift) {
                assert_eq!(x, i as u64 == shift, "Bit {} was wrong", i);
            }
        }
    }

    quickcheck! {
        fn bounds_check_get(x: Word) -> bool {
            x.get(x.len()).is_none()
        }

        fn test_set(x: Word) -> bool {
            (0..x.len()).all(|i| {
                let set_true = {
                    let mut x = x.clone();
                    x.set(i, true).unwrap();
                    x
                };
                let set_false = {
                    let mut x = x.clone();
                    x.set(i, false).unwrap();
                    x
                };

                (0..x.len()).all(|j| {
                    if i == j {
                        set_true.get(j).unwrap() == true
                        && set_false.get(j).unwrap() == false
                    } else {
                        set_true.get(j) == x.get(j)
                        && set_false.get(j) == x.get(j)
                    }
                })
            })
        }

        fn bounds_check_set(x: Word, to: bool) -> bool {
            let mut x = x;
            let len = x.len();
            x.set(len, to).is_none()
        }

        fn test_complement(x: Word) -> bool {
            let y = x.complement();
            (0..x.len()).all(|i| x.get(i) != y.get(i))
        }
    }

    quickcheck! {
        fn test_count(x: Word) -> bool {
            let bits = (0..64).map(|i| x.get(i).unwrap());
            let expected_count_ones = bits.clone().filter(|&x| x).count();
            let expected_count_zeros = bits.filter(|&x| !x).count();
            x.count_ones() == expected_count_ones as u32
                && x.count_zeros() == expected_count_zeros as u32
        }

        fn test_rank(x: Word) -> () {
            let mut expected_rank_ones = 0u32;
            let mut expected_rank_zeros = 0u32;
            for i in 0..x.len() {
                assert_eq!(Some(expected_rank_ones), x.rank_ones(i));
                assert_eq!(Some(expected_rank_zeros), x.rank_zeros(i));
                if x.get(i).unwrap() {
                    expected_rank_ones += 1;
                } else {
                    expected_rank_zeros += 1;
                }
            }
            assert_eq!(None, x.rank_ones(x.len()));
            assert_eq!(None, x.rank_zeros(x.len()));
        }

        fn test_select_u16(x: u16) -> () {
            let total_count = x.count_ones() as usize;
            let x_bits = Word::from((x as u64) << 48);
            for i in 0..total_count {
                let r = select_ones_u16(x, i as u32) as usize;
                let prev_rank = (0..r).filter(|&j| x_bits.get(j).unwrap()).count();
                assert_eq!(prev_rank, i);
                assert!(x_bits.get(r).unwrap());
            }
            assert_eq!(select_ones_u16(x, total_count as u32), 16);
            assert_eq!(select_ones_u16(x, 16), 16);
        }

        fn test_select_ones(x: Word) -> () {
            let total_count = x.count_ones() as usize;
            for i in 0..total_count {
                let r = x.select_ones(i as u32).unwrap() as usize;
                assert_eq!(Some(i as u32), x.rank_ones(r));
                assert_eq!(Some(true), x.get(r));
            }
            assert_eq!(None, x.select_ones(total_count as u32));
        }

        fn test_select_zeros(x: Word) -> () {
            let total_count = x.count_zeros() as usize;
            for i in 0..total_count {
                let r = x.select_zeros(i as u32).unwrap() as usize;
                assert_eq!(Some(i as u32), x.rank_zeros(r));
                assert_eq!(Some(false), x.get(r));
            }
            assert_eq!(None, x.select_zeros(total_count as u32));
        }
    }
}
