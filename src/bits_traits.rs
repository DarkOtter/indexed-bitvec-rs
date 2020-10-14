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

/// Supply to generic functions to work with/count set bits (ones).
pub(crate) enum OneBits {}

/// Supply to generic functions to work with/count unset bits (zeros).
pub(crate) enum ZeroBits {}

mod private {
    pub trait Sealed {}
    impl Sealed for super::OneBits {}
    impl Sealed for super::ZeroBits {}
}

/// Trait used to provide functionality generic across 1 or 0 bits,
/// for example rank in terms of set or unset bits.
pub(crate) trait OnesOrZeros: private::Sealed {
    /// Convert a count of ones in a range to a count of ones or zeros.
    /// The result is never larger than the number of bits supplied.
    /// It is assumed the count of ones is not larger than the number of bits.
    fn convert_count(count_ones: u64, in_bits: u64) -> u64;

    fn is_ones() -> bool;
}

impl OnesOrZeros for OneBits {
    #[inline]
    fn convert_count(count_ones: u64, _in_bits: u64) -> u64 {
        count_ones
    }

    #[inline]
    fn is_ones() -> bool {
        true
    }
}

impl OnesOrZeros for ZeroBits {
    #[inline]
    fn convert_count(count_ones: u64, in_bits: u64) -> u64 {
        in_bits - count_ones
    }

    #[inline]
    fn is_ones() -> bool {
        false
    }
}

pub trait Bits {
    fn len(&self) -> u64;
    fn get(&self, idx: u64) -> Option<bool>;
    fn count_ones(&self) -> u64;
    fn count_zeros(&self) -> u64 {
        use crate::import::prelude::*;
        ZeroBits::convert_count(self.count_ones(), self.len())
    }
    fn rank_ones(&self, idx: u64) -> Option<u64>;
    fn rank_zeros(&self, idx: u64) -> Option<u64> {
        use crate::import::prelude::*;
        let rank_ones = self.rank_ones(idx)?;
        Some(ZeroBits::convert_count(rank_ones, idx))
    }
    fn select_ones(&self, target_rank: u64) -> Option<u64>;
    fn select_zeros(&self, target_rank: u64) -> Option<u64>;
}

pub trait BitsSplit: Sized {
    fn split_at(self, mid: u64) -> Option<(Self, Self)>;
}

pub trait BitsMut {
    fn replace(&mut self, idx: u64, with: bool) -> Option<bool>;
    fn set(&mut self, idx: u64, to: bool) {
        match self.replace(idx, to) {
            Some(_) => (),
            None => panic!("Index is out of bounds for bits"),
        }
    }
}

pub trait BitsVec {
    fn push(&mut self, bit: bool);
}

#[cfg(test)]
pub mod tests {
    use super::*;

    fn sub_checked(a: u64, b: u64) -> u64 {
        a.checked_sub(b).expect("This subtraction should not overflow")
    }

    fn get_checked<B: Bits + ?Sized>(bits: &B, idx: u64) -> bool {
        bits.get(idx).expect("This index should not be out of bounds")
    }

    pub mod from_get_and_len {
        use super::*;

        pub fn test_count<B: Bits + ?Sized>(bits: &B) {
            let direct_count_ones =
                (0..bits.len()).map(|idx| get_checked(bits, idx) as u64).sum::<u64>();
            assert_eq!(bits.count_ones(), direct_count_ones);
            assert_eq!(bits.count_zeros(), sub_checked(bits.len(), direct_count_ones));
        }

        pub fn test_rank<B: Bits + ?Sized>(bits: &B) {
            let mut running_rank = 0;
            for idx in 0..bits.len() {
                assert_eq!(bits.rank_ones(idx), Some(running_rank));
                assert_eq!(bits.rank_zeros(idx), Some(sub_checked(idx, running_rank)));
                running_rank += get_checked(bits, idx) as u64;
            }
            assert!(bits.rank_ones(bits.len()).is_none());
            assert!(bits.rank_zeros(bits.len()).is_none());
        }

        pub fn test_select<B: Bits + ?Sized>(bits: &B) {
            let mut running_rank = 0;
            for idx in 0..bits.len() {
                let bit = get_checked(bits, idx);
                if bit {
                    assert_eq!(bits.select_ones(running_rank), Some(idx));
                } else {
                    assert_eq!(bits.select_zeros(sub_checked(idx, running_rank)), Some(idx));
                }
                running_rank += bit as u64;
            }
            assert!(bits.select_ones(running_rank).is_none());
            assert!(bits.select_zeros(sub_checked(bits.len(), running_rank)).is_none());
        }

        pub fn test_replace_at<B: Bits + BitsMut + ?Sized>(bits: &mut B, idx: u64) {
            let previous = match bits.get(idx) {
                Some(x) => x,
                None => {
                    assert!(bits.replace(idx, true).is_none());
                    assert!(bits.replace(idx, false).is_none());
                    return;
                },
            };

            assert_eq!(bits.replace(idx, true), Some(previous));
            assert_eq!(bits.get(idx), Some(true));
            assert_eq!(bits.replace(idx, false), Some(true));
            assert_eq!(bits.get(idx), Some(false));
            assert_eq!(bits.replace(idx, previous), Some(false));
            assert_eq!(bits.get(idx), Some(previous));
        }

        pub fn test_replace<B: Bits + BitsMut + ?Sized>(bits: &mut B) {
            (0..=bits.len()).for_each(|idx| test_replace_at(bits, idx))
        }

        pub fn test_set_at_in_range<B: Bits + BitsMut + ?Sized>(bits: &mut B, idx: u64) {
            if bits.get(idx).is_none() {
                panic!("This can't test what set does out of range");
            }

            bits.set(idx, true);
            assert_eq!(bits.get(idx), Some(true));
            bits.set(idx, false);
            assert_eq!(bits.get(idx), Some(false));
        }

        pub fn test_set_in_bounds<B: Bits + BitsMut + ?Sized>(bits: &mut B) {
            (0..bits.len()).for_each(|idx| test_set_at_in_range(bits, idx))
        }

        pub fn test_push<B: Bits + BitsVec + ?Sized>(bits: &mut B) {
            use oorandom::Rand64;
            let mut rng = Rand64::new(42);
            for _ in 0..1024 {
                let len_before = bits.len();
                assert!(bits.get(len_before).is_none());
                let set_to = (rng.rand_u64() % 2) == 0;
                bits.push(set_to);
                assert_eq!(bits.len(), len_before + 1);
                assert_eq!(bits.get(len_before), Some(set_to));
                assert!(bits.get(len_before + 1).is_none());
            }
        }
    }
}
