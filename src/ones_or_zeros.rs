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

/// Supply to generic functions to work with/count set bits (ones).
pub enum OneBits {}
/// Supply to generic functions to work with/count unset bits (zeros).
pub enum ZeroBits {}

mod private {
    pub trait Sealed {}
    impl Sealed for super::OneBits {}
    impl Sealed for super::ZeroBits {}
}

/// Trait used to provide functionality generic across 1 or 0 bits,
/// for example rank in terms of set or unset bits.
pub trait OnesOrZeros: private::Sealed {
    /// Convert a count of ones in a range to a count of ones or zeros.
    /// The result is never larger than the number of bits supplied.
    /// It is assumed the count of ones is not larger than the number of bits.
    fn convert_count(count_ones: u64, in_bits: u64) -> u64;

    fn is_ones() -> bool;

    /// Either count_ones or count_zeros.
    fn count(of_bits: u64) -> u32;
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

    #[inline]
    fn count(of_bits: u64) -> u32 {
        of_bits.count_ones()
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

    #[inline]
    fn count(of_bits: u64) -> u32 {
        of_bits.count_ones()
    }
}
