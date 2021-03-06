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

#[cfg(test)]
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

#[cold]
fn ceil_div_slow(n: usize, d: usize) -> usize {
    n / d + (if n % d > 0 { 1 } else { 0 })
}

#[inline(always)]
pub(crate) fn ceil_div(n: usize, d: usize) -> usize {
    let nb = n.wrapping_add(d - 1);
    if nb < n {
        return ceil_div_slow(n, d);
    };
    nb / d
}

#[cold]
fn ceil_div_u64_slow(n: u64, d: u64) -> u64 {
    n / d + (if n % d > 0 { 1 } else { 0 })
}

#[inline(always)]
pub(crate) fn ceil_div_u64(n: u64, d: u64) -> u64 {
    let nb = n.wrapping_add(d - 1);
    if nb < n {
        return ceil_div_u64_slow(n, d);
    };
    nb / d
}

mod ones_or_zeros;

mod word;

mod bytes;

pub mod bits_ref;

mod with_offset;

pub mod index_raw;

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn check_max_bits_in_bytes() {
        assert!(<u64>::max_value() / 8 <= <usize>::max_value() as u64);
    }

    #[test]
    fn test_ceil_div_examples() {
        assert_eq!(0, ceil_div(0, 4));
        assert_eq!(1, ceil_div(1, 4));
        assert_eq!(1, ceil_div(4, 4));
        assert_eq!(2, ceil_div(5, 4));
        assert_eq!(2, ceil_div(6, 4));
        assert_eq!(2, ceil_div(7, 4));
        assert_eq!(2, ceil_div(8, 4));

        assert_eq!(ceil_div_slow(0, 43), ceil_div(0, 43));
        assert_eq!(ceil_div_slow(1, 43), ceil_div(1, 43));
        assert_eq!(
            ceil_div_slow(usize::max_value(), 43),
            ceil_div(usize::max_value(), 43)
        );
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
        fn test_ceil_div(x in any::<usize>(), d in 1..999999usize) {
            prop_assert_eq!(ceil_div_slow(x, d), ceil_div(x, d));
        }

        #[test]
        fn test_ceil_div_64(x in any::<u64>(), d in 1..999999u64) {
            prop_assert_eq!(ceil_div_u64_slow(x, d), ceil_div_u64(x, d));
        }
    }
}
