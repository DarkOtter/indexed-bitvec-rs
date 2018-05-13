use super::ceil_div_u64;
use std::ops::Deref;
use std::cmp::min;
use bytes;
use ones_or_zeros::OnesOrZeros;

#[derive(Copy, Clone, Debug)]
pub struct Bits<T: Deref<Target = [u8]>>((T, u64));

fn big_enough(bytes: &[u8], used_bits: u64) -> bool {
    ceil_div_u64(used_bits, 8) <= bytes.len() as u64
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

    pub fn bytes(&self) -> &[u8] {
        let all_bytes = self.all_bytes();
        debug_assert!(big_enough(all_bytes, self.used_bits()));
        // This will not overflow because we checked the size
        // when we made self...
        &all_bytes[..ceil_div_u64(self.used_bits(), 8) as usize]
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

    pub fn count<W: OnesOrZeros>(&self) -> u64 {
        if (self.used_bits() % 8) != 0 {
            bytes::rank::<W>(self.all_bytes(), self.used_bits())
                .expect("Internally called rank out-of-range")
        } else {
            bytes::count::<W>(self.bytes())
        }
    }

    pub fn rank<W: OnesOrZeros>(&self, idx: u64) -> Option<u64> {
        if idx >= self.used_bits() {
            None
        } else {
            Some(bytes::rank::<W>(self.all_bytes(), idx).expect(
                "Internally called rank out-of-range",
            ))
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

    pub(crate) fn chunks_bytes(&self, bytes_per_chunk: usize) -> impl Iterator<Item = Bits<&[u8]>> {
        let available_bits = self.used_bits();
        let bits_per_chunk = (bytes_per_chunk as u64) * 8;
        self.bytes().chunks(bytes_per_chunk).enumerate().map(
            move |(i, chunk)| {
                let used_bits = i as u64 * bits_per_chunk;
                let bits = min(available_bits - used_bits, bits_per_chunk);
                Bits::from(chunk, bits).expect("Size invariant violated")
            },
        )
    }

    pub(crate) fn skip_bytes(&self, idx_bytes: usize) -> Bits<&[u8]> {
        let bytes = self.bytes();
        if idx_bytes >= bytes.len() {
            panic!("Tried to skip the whole of a bitvector");
        }
        Bits::from(
            &bytes[idx_bytes..],
            self.used_bits() - (idx_bytes as u64 * 8),
        ).expect("Checked sufficient bytes are present")
    }

    pub fn clone_ref(&self) -> Bits<&[u8]> {
        Bits::from(self.all_bytes(), self.used_bits()).expect("Previously checked")
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
        assert_eq!(12, make(16).count::<OneBits>());
        assert_eq!(4, make(16).count::<ZeroBits>());
        assert_eq!(12, make(15).count::<OneBits>());
        assert_eq!(3, make(15).count::<ZeroBits>());
        assert_eq!(11, make(14).count::<OneBits>());
        assert_eq!(3, make(14).count::<ZeroBits>());
        assert_eq!(11, make(13).count::<OneBits>());
        assert_eq!(2, make(13).count::<ZeroBits>());
        assert_eq!(10, make(12).count::<OneBits>());
        assert_eq!(2, make(12).count::<ZeroBits>());
        assert_eq!(10, make(11).count::<OneBits>());
        assert_eq!(1, make(11).count::<ZeroBits>());
        assert_eq!(9, make(10).count::<OneBits>());
        assert_eq!(1, make(10).count::<ZeroBits>());
        assert_eq!(9, make(9).count::<OneBits>());
        assert_eq!(0, make(9).count::<ZeroBits>());
        assert_eq!(8, make(8).count::<OneBits>());
        assert_eq!(0, make(8).count::<ZeroBits>());
        assert_eq!(7, make(7).count::<OneBits>());
        assert_eq!(0, make(7).count::<ZeroBits>());
        assert_eq!(0, make(0).count::<OneBits>());
        assert_eq!(0, make(0).count::<ZeroBits>());
    }

    #[test]
    fn test_rank() {
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| Bits::from(bytes_a, len).expect("valid");
        let bits_a = make(16);
        for i in 0..15 {
            assert_eq!(Some(make(i).count::<OneBits>()), bits_a.rank::<OneBits>(i));
            assert_eq!(
                Some(make(i).count::<ZeroBits>()),
                bits_a.rank::<ZeroBits>(i)
            );
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
