use super::MAX_BITS;
use std::ops::Deref;
use bytes;
use ones_or_zeros::OnesOrZeros;

#[derive(Copy, Clone, Debug)]
pub struct Bits<T: Deref<Target = [u8]>>((T, u64));

fn used_bits_to_bytes(bits: u64) -> usize {
    if bits > MAX_BITS {
        panic!("Too many bits");
    }
    let whole_bytes = (bits >> 3) as usize;
    let partial_byte = {
        let x = (bits & 7) as usize;
        (x >> 2 | x >> 1 | x) & 1
    };
    whole_bytes + partial_byte
}

fn big_enough(bytes: &[u8], used_bits: u64) -> bool {
    if used_bits > MAX_BITS {
        false
    } else {
        bytes.len() >= used_bits_to_bytes(used_bits)
    }
}

impl<T: Deref<Target = [u8]>> Bits<T> {
    pub fn from(bytes: T, used_bits: u64) -> Option<Self> {
        if big_enough(bytes.deref(), used_bits) {
            Some(Bits((bytes, used_bits)))
        } else {
            None
        }
    }

    pub fn all_bytes(&self) -> &[u8] {
        (self.0).0.deref()
    }

    pub fn used_bits(&self) -> u64 {
        (self.0).1
    }

    fn bytes_for(&self, used_bits: u64) -> &[u8] {
        let all_bytes = self.all_bytes();
        debug_assert!(big_enough(all_bytes, used_bits));
        &all_bytes[..used_bits_to_bytes(used_bits)]
    }

    pub fn bytes(&self) -> &[u8] {
        self.bytes_for(self.used_bits())
    }

    pub fn decompose(self) -> (T, u64) {
        self.0
    }

    pub fn get(&self, idx_bits: u64) -> Option<bool> {
        if idx_bits >= self.used_bits() {
            None
        } else {
            debug_assert!(big_enough(self.all_bytes(), self.used_bits()));
            Some(bytes::get_unchecked(self.all_bytes(), idx_bits))
        }
    }

    fn count_part<W: OnesOrZeros>(&self, used_bits: u64) -> Option<u64> {
        let bytes = self.bytes_for(used_bits);
        if bytes.len() == 0 {
            return Some(0);
        };
        let whole_bytes_count = bytes::count::<W>(bytes)?;
        let rem = used_bits & 7;
        if rem == 0 {
            return Some(whole_bytes_count);
        };
        let last_byte = bytes[bytes.len() - 1];
        let last_byte = if W::is_ones() { last_byte } else { !last_byte };
        let mask = !0u8 >> rem;
        Some(whole_bytes_count - (last_byte & mask).count_ones() as u64)
    }

    pub fn count<W: OnesOrZeros>(&self) -> Option<u64> {
        self.count_part::<W>(self.used_bits())
    }

    pub fn rank<W: OnesOrZeros>(&self, idx: u64) -> Option<u64> {
        if idx >= self.used_bits() {
            None
        } else {
            self.count_part::<W>(idx)
        }
    }

    pub fn select<W: OnesOrZeros>(&self, target_rank: u64) -> Option<u64> {
        let res = bytes::select::<W>(self.bytes(), target_rank);
        match res {
            None => None,
            Some(res) => {
                if res >= self.used_bits() {
                    None
                } else {
                    Some(res)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck;
    use quickcheck::Arbitrary;
    use ones_or_zeros::{OneBits, ZeroBits};

    impl Arbitrary for Bits<Box<[u8]>> {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            let data = <Vec<u8>>::arbitrary(g);
            let all_bits = data.len() as u64 * 8;
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

    #[test]
    fn test_count() {
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| Bits::from(bytes_a, len).expect("valid");
        assert_eq!(Some(12), make(16).count::<OneBits>());
        assert_eq!(Some(4), make(16).count::<ZeroBits>());
        assert_eq!(Some(12), make(15).count::<OneBits>());
        assert_eq!(Some(3), make(15).count::<ZeroBits>());
        assert_eq!(Some(11), make(14).count::<OneBits>());
        assert_eq!(Some(3), make(14).count::<ZeroBits>());
        assert_eq!(Some(11), make(13).count::<OneBits>());
        assert_eq!(Some(2), make(13).count::<ZeroBits>());
        assert_eq!(Some(10), make(12).count::<OneBits>());
        assert_eq!(Some(2), make(12).count::<ZeroBits>());
        assert_eq!(Some(10), make(11).count::<OneBits>());
        assert_eq!(Some(1), make(11).count::<ZeroBits>());
        assert_eq!(Some(9), make(10).count::<OneBits>());
        assert_eq!(Some(1), make(10).count::<ZeroBits>());
        assert_eq!(Some(9), make(9).count::<OneBits>());
        assert_eq!(Some(0), make(9).count::<ZeroBits>());
        assert_eq!(Some(8), make(8).count::<OneBits>());
        assert_eq!(Some(0), make(8).count::<ZeroBits>());
        assert_eq!(Some(7), make(7).count::<OneBits>());
        assert_eq!(Some(0), make(7).count::<ZeroBits>());
        assert_eq!(Some(0), make(0).count::<OneBits>());
        assert_eq!(Some(0), make(0).count::<ZeroBits>());
    }

    #[test]
    fn test_rank() {
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| Bits::from(bytes_a, len).expect("valid");
        let bits_a = make(16);
        for i in 0..15 {
            assert!(bits_a.rank::<OneBits>(i).is_some());
            assert!(bits_a.rank::<ZeroBits>(i).is_some());
            assert_eq!(make(i).count::<OneBits>(), bits_a.rank::<OneBits>(i));
            assert_eq!(make(i).count::<ZeroBits>(), bits_a.rank::<ZeroBits>(i));
        }
        assert_eq!(None, bits_a.rank::<OneBits>(16));
        assert_eq!(None, bits_a.rank::<ZeroBits>(16));
        assert_eq!(None, make(13).rank::<OneBits>(13));
        assert_eq!(None, make(13).rank::<ZeroBits>(13));
        assert_eq!(bits_a.rank::<OneBits>(12), make(13).rank::<OneBits>(12));
        assert_eq!(bits_a.rank::<ZeroBits>(12), make(13).rank::<ZeroBits>(12));

    }

    #[test]
    fn test_select() {
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| Bits::from(bytes_a, len).expect("valid");
        assert_eq!(Some(14), make(16).select::<OneBits>(11));
        assert_eq!(None, make(14).select::<OneBits>(11));
    }
}
