use std::ops::Deref;

#[derive(Copy, Clone, Debug)]
pub struct Bits<T: Deref<Target = [u8]>>((T, usize));

fn valid_size(bytes: &[u8], used_bits: usize) -> bool {
    let used_bits_in_bytes = used_bits * 8;
    used_bits_in_bytes <= bytes.len() && bytes.len() < used_bits_in_bytes + 8
}

impl<T: Deref<Target = [u8]>> Bits<T> {
    pub fn from(bytes: T, used_bits: usize) -> Option<Self> {
        if valid_size(bytes.deref(), used_bits) {
            Some(Bits((bytes, used_bits)))
        } else {
            None
        }
    }

    pub fn bytes(&self) -> &[u8] {
        (self.0).0.deref()
    }

    pub fn used_bits(&self) -> usize {
        let used_bits = (self.0).1;
        debug_assert!(valid_size(self.bytes(), (self.0).1));
        used_bits
    }
}
