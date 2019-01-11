#[cfg(test)]
mod tests {
    use indexed_bitvec_core::bits::*;
    use proptest::collection::vec as gen_vec;
    use proptest::collection::SizeRange;
    use proptest::prelude::*;

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

    fn test_count_via_get(bits: Bits<Vec<u8>>, bit_to_count: bool) -> Result<(), TestCaseError> {
        fn inner<F>(bits: Bits<Vec<u8>>, bit_to_count: bool, f: F) -> Result<(), TestCaseError>
        where
            F: Fn(&Bits<Vec<u8>>) -> u64,
        {
            let count_via_get = (0..bits.used_bits())
                .filter(|&idx| bits.get(idx).unwrap() == bit_to_count)
                .count() as u64;
            prop_assert_eq!(count_via_get, f(&bits));
            Ok(())
        }

        if bit_to_count {
            inner(bits, true, Bits::count_ones)
        } else {
            inner(bits, false, Bits::count_zeros)
        }
    }

    proptest! {
        #[test]
        fn test_count_ones_via_get(bits in gen_bits(0..=1024)) {
            test_count_via_get(bits, true)?;
        }

        #[test]
        fn test_count_ones_via_iter(bits in gen_bits(0..=1024)) {
            let count_via_iter =
                bits.iter()
                .filter(|&b| b)
                .count() as u64;
            prop_assert_eq!(count_via_iter, bits.count_ones());

        }

        #[test]
        fn test_count_zeros_via_get(bits in gen_bits(0..=1024)) {
            test_count_via_get(bits, false)?;
        }

        #[test]
        fn test_count_zeros_via_count_ones(bits in gen_bits(0..=1024)) {
            prop_assert_eq!(bits.used_bits() - bits.count_ones(), bits.count_zeros());
        }

    }

    // TODO: Test rank via count instead (make the bits smaller)
    fn test_rank_via_get(bits: Bits<Vec<u8>>, bit_to_rank: bool) -> Result<(), TestCaseError> {
        fn inner<F>(bits: Bits<Vec<u8>>, bit_to_rank: bool, f: F) -> Result<(), TestCaseError>
        where
            F: Fn(&Bits<Vec<u8>>, u64) -> Option<u64>,
        {
            let mut running_rank = 0;
            for idx in 0..=(bits.used_bits() + 64) {
                if idx >= bits.used_bits() {
                    prop_assert_eq!(None, f(&bits, idx), "should be out of range at {}", idx);
                } else {
                    prop_assert_eq!(
                        Some(running_rank),
                        f(&bits, idx),
                        "disagree at index {}",
                        idx
                    );
                    if bits.get(idx).unwrap() == bit_to_rank {
                        running_rank += 1;
                    }
                }
            }
            Ok(())
        }

        if bit_to_rank {
            inner(bits, true, Bits::rank_ones)
        } else {
            inner(bits, false, Bits::rank_zeros)
        }
    }

    proptest! {
        #[test]
        fn test_rank_ones_via_get(bits in gen_bits(0..=1024)) {
            test_rank_via_get(bits, true)?;
        }

        #[test]
        fn test_rank_ones_via_iter(bits in gen_bits(0..=1024)) {
            let mut idx = 0u64;
            let mut running_rank_ones = 0;
            for b in bits.iter() {
                prop_assert_eq!(Some(running_rank_ones), bits.rank_ones(idx),
                                "disagree at index {}", idx);
                idx += 1;
                if b { running_rank_ones += 1 };
            }
            prop_assert_eq!(None, bits.rank_ones(idx),
                            "should be out of range at {}", idx);
        }

        #[test]
        fn test_rank_zeros_via_get(bits in gen_bits(0..=1024)) {
            test_rank_via_get(bits, false)?;
        }

        #[test]
        fn test_rank_zeros_via_rank_ones(bits in gen_bits(0..=1024)) {
            for idx in 0..=(bits.used_bits() + 64) {
                let via_rank_ones =
                    bits.rank_ones(idx).map(|ones| idx - ones);
                prop_assert_eq!(via_rank_ones, bits.rank_zeros(idx));
            }
        }
    }

    fn test_select_via_count_rank_get(
        bits: Bits<Vec<u8>>,
        bit_to_select: bool,
    ) -> Result<(), TestCaseError> {
        fn inner<C, R, S>(
            bits: Bits<Vec<u8>>,
            bit_to_select: bool,
            count: C,
            rank: R,
            select: S,
        ) -> Result<(), TestCaseError>
        where
            C: Fn(&Bits<Vec<u8>>) -> u64,
            R: Fn(&Bits<Vec<u8>>, u64) -> Option<u64>,
            S: Fn(&Bits<Vec<u8>>, u64) -> Option<u64>,
        {
            for bit_idx in 0..count(&bits) {
                let select_idx = select(&bits, bit_idx);
                prop_assert!(
                    select_idx.is_some(),
                    "expected bit_idx to exist: {}",
                    bit_idx
                );
                let select_idx = select_idx.unwrap();
                prop_assert_eq!(
                    Some(bit_idx),
                    rank(&bits, select_idx),
                    "expected rank to be {} at {}",
                    bit_idx,
                    select_idx
                );
                prop_assert_eq!(
                    Some(bit_to_select),
                    bits.get(select_idx),
                    "expected bit to be {} at {}",
                    bit_to_select,
                    select_idx
                );
                // TODO: Finish
            }

            prop_assert_eq!(
                None,
                select(&bits, count(&bits)),
                "expected no selected rank for count"
            );

            Ok(())
        }

        if bit_to_select {
            inner(
                bits,
                true,
                Bits::count_ones,
                Bits::rank_ones,
                Bits::select_ones,
            )
        } else {
            inner(
                bits,
                false,
                Bits::count_zeros,
                Bits::rank_zeros,
                Bits::select_zeros,
            )
        }
    }

    proptest! {
        #[test]
        fn test_select_ones_via_count_rank_get(bits in gen_bits(0..=1024)) {
            test_select_via_count_rank_get(bits, true)?;
        }

        #[test]
        fn test_select_zeros_via_count_rank_get(bits in gen_bits(0..=1024)) {
            test_select_via_count_rank_get(bits, false)?;
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

    proptest! {
        #[test]
        fn test_iter_and_into_iter_via_get(bits in gen_bits(0..=1024)) {
            let from_get: Vec<_> = (0..bits.used_bits())
                .map(|idx| bits.get(idx).unwrap())
                .collect();
            let from_iter: Vec<_> = bits.iter().collect();
            let from_into_iter: Vec<_> = bits.into_iter().collect();
            prop_assert_eq!(&from_get, &from_iter);
            prop_assert_eq!(&from_get, &from_into_iter);
        }

        #[test]
        fn test_iter_and_into_iter_set_bits_via_get(bits in gen_bits(0..=1024)) {
            let from_get: Vec<_> = (0..bits.used_bits())
                .filter(|&idx| bits.get(idx).unwrap())
                .collect();
            let from_iter: Vec<_> = bits.iter_set_bits().collect();
            let from_into_iter: Vec<_> = bits.into_iter_set_bits().collect();
            prop_assert_eq!(&from_get, &from_iter);
            prop_assert_eq!(&from_get, &from_into_iter);
        }

        #[test]
        fn test_iter_and_into_iter_zero_bits_via_get(bits in gen_bits(0..=1024)) {
            let from_get: Vec<_> = (0..bits.used_bits())
                .filter(|&idx| !(bits.get(idx).unwrap()))
                .collect();
            let from_iter: Vec<_> = bits.iter_zero_bits().collect();
            let from_into_iter: Vec<_> = bits.into_iter_zero_bits().collect();
            prop_assert_eq!(&from_get, &from_iter);
            prop_assert_eq!(&from_get, &from_into_iter);
        }
    }

    fn test_eq_and_cmp_on(l: Bits<Vec<u8>>, r: Bits<Vec<u8>>) -> Result<(), TestCaseError> {
        let l_as_vec: Vec<_> = l.iter().collect();
        let r_as_vec: Vec<_> = r.iter().collect();
        prop_assert_eq!(l_as_vec.eq(&r_as_vec), l.eq(&r));
        prop_assert_eq!(l_as_vec.cmp(&r_as_vec), l.cmp(&r));
        Ok(())
    }

    proptest! {
        #[test]
        fn test_eq_and_cmp_pair(l in gen_bits(0..=1024), r in gen_bits(0..=1024)) {
            test_eq_and_cmp_on(l, r)?;
        }

        #[test]
        fn test_eq_and_cmp_single(x in gen_bits(0..=1024)) {
            let y = x.clone();
            test_eq_and_cmp_on(x, y)?;
        }
    }

    proptest! {
        #[test]
        fn test_serialise_roundtrip(original in gen_bits(0..=1024)) {
            let serialised = bincode::serialize(&original).unwrap();
            let deserialised = bincode::deserialize(&serialised).unwrap();
            prop_assert_eq!(original, deserialised);
        }
    }
}
