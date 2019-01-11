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
//! Operations to create indexes used to perform
//! fast rank & select operations on bitvectors.

extern crate serde;
#[macro_use]
extern crate serde_derive;

extern crate indexed_bitvec_core;

#[cfg(feature = "implement_heapsize")]
extern crate heapsize;

#[cfg(test)]
#[macro_use]
extern crate proptest;
#[cfg(test)]
extern crate bincode;

mod bits;
pub use indexed_bitvec_core::bits::Bits;

mod indexed_bits;
pub use crate::indexed_bits::IndexedBits;

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

// TODO: Test ceil_div
