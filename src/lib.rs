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
//! Core operations to create indexes used to perform
//! fast rank & select operations on bitvectors.
#![no_std]

#[cfg(any(test, feature = "std"))]
#[macro_use]
extern crate std;

#[cfg(all(feature = "alloc", not(any(test, feature = "std"))))]
extern crate alloc;

#[macro_use]
extern crate static_assertions;

#[cfg(test)]
extern crate proptest;

mod bits_traits;

mod import {
    pub mod prelude {
        pub use core::ops::Range;
        pub use core::mem::{swap, replace, align_of, size_of};
        pub(crate) use crate::bits_traits::{OnesOrZeros, ZeroBits};
        pub use crate::bits_traits::{Bits, BitsMut, BitsSplit, BitsVec};
        pub use core::cmp::{min, max, Ordering};

        #[inline]
        pub const fn ceil_div_u64(n: u64, d: u64) -> u64 {
            n / d + ((n % d > 0) as u64)
        }

        #[cfg(any(test, feature = "std"))]
        pub use std::vec::Vec;
        #[cfg(all(feature = "alloc", not(any(test, feature = "std"))))]
        pub use alloc::vec::Vec;

        #[cfg(any(test, feature = "std"))]
        pub use std::boxed::Box;
        #[cfg(all(feature = "alloc", not(any(test, feature = "std"))))]
        pub use alloc::boxed::Box;
    }

    pub use core::slice;
    pub use core::iter;
    pub use core::ops::{Deref, DerefMut};
    pub use core::default::Default;
    pub use core::borrow::{Borrow, BorrowMut};
}

mod word;
pub use word::Word;

pub mod bits;
pub mod index;

// TODO: Force documentation
// TODO: Look at adding doctests
// TODO: Review public interface

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
    }

    proptest! {
        #[test]
        fn test_ceil_div_64(n in any::<u64>(), d in 1..999999u64) {
            let floor_div = n / d;
            let ceil_div = ceil_div_u64(n, d);
            if floor_div == ceil_div {
                assert_eq!(ceil_div * d, n);
            } else {
                assert_eq!(floor_div + 1, ceil_div);
                assert!(floor_div * d < n);
                assert!(ceil_div * d > n);
            }
        }
    }
}
