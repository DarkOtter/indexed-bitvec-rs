use std::ops::Deref;
use super::ceil_div;

fn used_bits_to_bytes(bits: usize) -> usize {
    let whole_bytes = bits >> 3;
    let partial_byte = {
        let x = bits & 7;
        (x >> 2 | x >> 1 | x) & 1
    };
    whole_bytes + partial_byte
}

#[derive(Copy, Clone, Debug)]
pub struct Bits<T: Deref<Target = [u8]>>((T, usize));

fn big_enough(bytes: &[u8], used_bits: usize) -> bool {
    bytes.len() >= ceil_div(used_bits, 8)
}

impl<T: Deref<Target = [u8]>> Bits<T> {
    pub fn from(bytes: T, used_bits: usize) -> Option<Self> {
        if big_enough(bytes.deref(), used_bits) {
            Some(Bits((bytes, used_bits)))
        } else {
            None
        }
    }

    pub fn used_bits(&self) -> usize {
        (self.0).1
    }

    pub fn bytes(&self) -> &[u8] {
        let all_bytes = (self.0).0.deref();
        debug_assert!(big_enough(all_bytes, self.used_bits()));
        &all_bytes[..used_bits_to_bytes(self.used_bits())]
    }

    pub fn decompose(self) -> (T, usize) {
        self.0
    }
}
