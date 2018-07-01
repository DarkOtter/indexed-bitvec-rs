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
//!
//! This crate is still under heavy development,
//! so it will not be very stable in its interface yet.

extern crate indexed_bitvec_core;

pub use indexed_bitvec_core::ones_or_zeros;
pub use indexed_bitvec_core::{OneBits, ZeroBits};
pub use indexed_bitvec_core::Bits;

mod indexed_bits;
pub use indexed_bits::IndexedBits;

/*
#[cfg(test)]
extern crate rand;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;
*/
