pub enum OneBits {}
pub enum ZeroBits {}

mod private {
    pub trait Sealed {}
    impl Sealed for super::OneBits {}
    impl Sealed for super::ZeroBits {}
}

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
