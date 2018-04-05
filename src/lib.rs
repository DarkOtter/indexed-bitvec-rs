extern crate byteorder;

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

pub mod bits64;
mod indexable;
pub mod bitvec64;
pub mod index;
