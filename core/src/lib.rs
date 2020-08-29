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
//! Core operations to create indexes used to perform
//! fast rank & select operations on bitvectors.
#![no_std]

#[cfg(any(test, feature = "std"))]
#[macro_use]
extern crate std;

#[cfg(test)]
extern crate rand;
#[cfg(test)]
extern crate rand_xorshift;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cfg(test)]
extern crate proptest;

mod import {
    pub mod prelude {
        pub use core::ops::Range;
        pub use core::mem::{replace, align_of, size_of};
        pub use crate::ones_or_zeros::{OnesOrZeros, OneBits, ZeroBits};
        pub use core::cmp::{min, max};

        #[cold]
        pub const fn ceil_div_u64_slow(n: u64, d: u64) -> u64 {
            n / d + ((n % d > 0) as u64)
        }

        #[inline(always)]
        pub fn ceil_div_u64(n: u64, d: u64) -> u64 {
            let nb = n.wrapping_add(d - 1);
            if nb < n {
                return ceil_div_u64_slow(n, d);
            };
            nb / d
        }
    }

    pub use core::slice;
    pub use core::iter;

    #[cfg(any(test, feature = "std"))]
    pub use std::vec::Vec;
    #[cfg(all(feature = "alloc", not(any(test, feature = "std"))))]
    pub use alloc::vec::Vec;

    #[cfg(any(test, feature = "std"))]
    pub use std::boxed::Box;
    #[cfg(all(feature = "alloc", not(any(test, feature = "std"))))]
    pub use alloc::boxed::Box;

    pub use core::ops::Deref;
}


mod ones_or_zeros;

trait Bits {
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
    fn select_ones(&self, idx: u64) -> Option<u64>;
    fn select_zeros(&self, idx: u64) -> Option<u64>;
}

trait BitsSplit {
    fn split_at(self, mid: u64) -> Option<(Self, Self)>;
}

trait BitsMut {
    fn replace(&mut self, idx: u64, with: bool) -> bool;
    fn set(&mut self, idx: u64, to: bool) {
        self.replace(idx, to);
    }
}

trait BitsSplitMut {
    fn split_at_mut(self, mid: u64) -> Option<(Self, Self)>;
}

trait BitsVec {
    fn push(&mut self, bit: bool);
}

mod word;

pub mod bits;

pub mod index_raw;

#[cfg(test)]
mod tests {
    use super::*;
    use import::prelude::*;
    use proptest::prelude::*;

    #[test]
    fn check_max_bits_in_bytes() {
        assert!(<u64>::max_value() / 8 <= <usize>::max_value() as u64);
    }

    #[test]
    fn test_ceil_div_u64_examples() {
        assert_eq!(0, ceil_div_u64(0, 4));
        assert_eq!(1, ceil_div_u64(1, 4));
        assert_eq!(1, ceil_div_u64(4, 4));
        assert_eq!(2, ceil_div_u64(5, 4));
        assert_eq!(2, ceil_div_u64(6, 4));
        assert_eq!(2, ceil_div_u64(7, 4));
        assert_eq!(2, ceil_div_u64(8, 4));

        assert_eq!(ceil_div_u64_slow(0, 43), ceil_div_u64(0, 43));
        assert_eq!(ceil_div_u64_slow(1, 43), ceil_div_u64(1, 43));
        assert_eq!(
            ceil_div_u64_slow(u64::max_value(), 43),
            ceil_div_u64(u64::max_value(), 43)
        );
    }

    proptest! {
        #[test]
        fn test_ceil_div_64(x in any::<u64>(), d in 1..999999u64) {
            prop_assert_eq!(ceil_div_u64_slow(x, d), ceil_div_u64(x, d));
        }
    }
}
