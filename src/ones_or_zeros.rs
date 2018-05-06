pub enum OneBits {}
pub enum ZeroBits {}

mod private {
    pub trait Sealed {}
    impl Sealed for super::OneBits {}
    impl Sealed for super::ZeroBits {}
}

pub trait OnesOrZeros {
    // Convert a count of ones in a range to a count of ones or zeros.
    // The result is never larger than the number of bits supplied.
    // It is assumed the count of ones is not larger than the number of bits.
    fn convert_count(count_ones: u64, in_bits: u64) -> u64;

    fn is_ones() -> bool;
}

impl OnesOrZeros for OneBits {
    fn convert_count(count_ones: u64, in_bits: u64) -> u64 {
        count_ones
    }

    fn is_ones() -> bool {
        true
    }
}

impl OnesOrZeros for ZeroBits {
    fn convert_count(count_ones: u64, in_bits: u64) -> u64 {
        in_bits - count_ones
    }

    fn is_ones() -> bool {
        false
    }
}
