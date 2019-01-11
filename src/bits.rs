#[cfg(test)]
mod tests {
    use indexed_bitvec_core::bits::*;
    use proptest::collection::vec as gen_vec;
    use proptest::collection::SizeRange;
    use proptest::prelude::*;

    #[test]
    fn test_basic_from() {
        let example_data = vec![0xff, 0xf0];
        for i in 0..=16 {
            assert!(Bits::from(example_data.clone(), i).is_some());
        }
        for i in 17..32 {
            assert!(Bits::from(example_data.clone(), i).is_none());
        }
    }

    #[test]
    fn test_basic_get() {
        let example_data = vec![0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];
        let max_len = 8 * 8;
        for len in 0..=max_len {
            let bits = Bits::from(example_data.clone(), len).unwrap();
            for i in 0..len {
                assert_eq!(Some(i / 8 == i % 8), bits.get(i));
            }
            for i in len..=(max_len + 1) {
                assert_eq!(None, bits.get(i));
            }
        }
    }

    #[test]
    fn test_basic_set() {
        let example_data = vec![0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];
        let max_len = 8 * 8;
        for len in 0..=max_len {
            let bits = Bits::from(example_data.clone(), len).unwrap();
            for set_at in 0..len {
                for set_to in vec![true, false].into_iter() {
                    let mut bits = bits.clone();
                    assert!(bits.set(set_at, set_to).is_ok());
                    let bits = bits;
                    for i in 0..len {
                        if i == set_at {
                            assert_eq!(Some(set_to), bits.get(i));
                        } else {
                            assert_eq!(Some(i / 8 == i % 8), bits.get(i));
                        }
                    }
                    for i in len..=(max_len + 1) {
                        assert_eq!(None, bits.get(i));
                    }
                }
            }
            let mut bits = bits;
            for set_at in len..=(max_len + 1) {
                assert!(bits.set(set_at, true).is_err());
                assert!(bits.set(set_at, false).is_err());
            }
        }
    }

    prop_compose! {
        fn gen_bits(byte_len: impl Into<SizeRange>)
            (data in gen_vec(any::<u8>(), byte_len))
            (used_bits in 0..=((data.len() as u64) * 8),
             data in Just(data))
            -> Bits<Vec<u8>>
        {
            Bits::from(data, used_bits).unwrap()
        }
    }

    proptest! {
        #[test]
        fn test_all_bytes(bits in gen_bits(0..=1024)) {
            let clone_of_data = bits.clone().decompose().0;
            prop_assert_eq!(&clone_of_data[..], bits.all_bytes());
        }

        #[test]
        fn test_bytes(bits in gen_bits(0..=1024)) {
            let need_bytes = ((bits.used_bits() + 7) / 8) as usize;
            prop_assert_eq!(&bits.all_bytes()[..need_bytes], bits.bytes());
        }
    }

    // TODO: Retest bits
    // TODO: Test serialisation
}
