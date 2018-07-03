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
//!
//! This crate is still under heavy development,
//! so it will not be very stable in its interface yet.
#![no_std]

#[cfg(test)]
#[macro_use]
extern crate std;

#[cfg(test)]
extern crate rand;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[cold]
fn ceil_div_slow(n: usize, d: usize) -> usize {
    n / d + (if n % d > 0 { 1 } else { 0 })
}

#[allow(dead_code)]
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

#[allow(dead_code)]
#[inline(always)]
pub(crate) fn ceil_div_u64(n: u64, d: u64) -> u64 {
    let nb = n.wrapping_add(d - 1);
    if nb < n {
        return ceil_div_u64_slow(n, d);
    };
    nb / d
}

pub mod ones_or_zeros;
pub use ones_or_zeros::{OneBits, ZeroBits};

mod word;
pub use word::Word;

mod bytes;

mod bits;
pub use bits::Bits;

mod with_offset;

pub mod parallelism_generic;
pub mod index_raw;

#[cfg(test)]
mod tests {
    #[test]
    fn check_max_bits_in_bytes() {
        assert!(<u64>::max_value() / 8 <= <usize>::max_value() as u64);
    }
}
