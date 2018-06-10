extern crate byteorder;

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

pub mod result;
pub mod word;
pub mod bytes;
pub mod bits_type;
pub mod index_raw;

#[cfg(test)]
mod tests {
    #[test]
    fn check_max_bits_in_bytes() {
        assert!(<u64>::max_value() / 8 <= <usize>::max_value() as u64);
    }
}
