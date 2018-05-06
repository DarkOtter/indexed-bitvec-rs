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

    pub fn get(&self, idx: usize) -> Option<bool> {
        if idx >= self.used_bits() {
            return None;
        }

        let byte_idx = idx >> 3;
        let idx_in_byte = idx & 7;

        let byte = self.bytes()[byte_idx];
        let mask = 0x80 >> idx_in_byte;
        Some((byte & mask) != 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck;
    use quickcheck::Arbitrary;

    impl Arbitrary for Bits<Box<[u8]>> {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            let data = <Vec<u8>>::arbitrary(g);
            let all_bits = data.len() * 8;
            let overflow = g.gen_range(0, 64);
            Self::from(data.into_boxed_slice(), all_bits.saturating_sub(overflow))
                .expect("Generated bits must be valid")
        }
    }

    #[test]
    fn test_get() {
        let pattern_a = [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];
        let bits_a = Bits::from(&pattern_a[..], 8 * 8).unwrap();
        for i in 0..bits_a.used_bits() {
            assert_eq!(
                bits_a.get(i).unwrap(),
                i / 8 == i % 8,
                "Differed at position {}",
                i
            )
        }

        let pattern_b = [0xff, 0xc0];
        let bits_b = Bits::from(&pattern_b[..], 10).unwrap();
        for i in 0..10 {
            assert_eq!(bits_b.get(i), Some(true), "Differed at position {}", i)
        }
        for i in 10..16 {
            assert_eq!(bits_b.get(i), None, "Differed at position {}", i)
        }
    }
}
