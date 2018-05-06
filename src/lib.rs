extern crate byteorder;

#[cfg(test)]
#[macro_use]
extern crate quickcheck;

#[inline(always)]
pub(crate) fn mult_64(i: usize) -> usize {
    i << 6
}

#[inline(always)]
pub(crate) fn div_64(i: usize) -> usize {
    i >> 6
}

#[inline(always)]
pub(crate) fn mod_64(i: usize) -> usize {
    i & 63
}

pub(crate) fn ceil_div_64(i: usize) -> usize {
    div_64(i) + (if mod_64(i) > 0 { 1 } else { 0 })
}

pub(crate) fn ceil_div(n: usize, d: usize) -> usize {
    n / d + (if n % d > 0 { 1 } else { 0 })
}

pub const MAX_BITS_IN_BYTES: usize = (<u64>::max_value() / 8) as usize;
pub const MAX_BITS: u64 = MAX_BITS_IN_BYTES as u64 * 8;

pub mod ones_or_zeros;
pub use ones_or_zeros::{OneBits, ZeroBits};

pub mod word;
mod bits_type;
pub mod bytes;
// pub mod bitvec64;
// pub mod index;

#[cfg(test)]
mod tests {
    #[test]
    fn check_max_bits_in_bytes() {
        assert!(<u64>::max_value() / 8 <= <usize>::max_value() as u64);
    }
}
