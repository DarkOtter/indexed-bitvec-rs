/*
   Copyright 2018 DarkOtter

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
//! A bitvector with an index to allow fast rank and select.
use crate::bits::Bits;
use indexed_bitvec_core::bits_ref::BitsRef;
use indexed_bitvec_core::index_raw;
use std::ops::Deref;

/// Bits stored with extra index data for fast rank and select.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct IndexedBits<T: Deref<Target = [u8]>> {
    index: Box<[u64]>,
    bits: Bits<T>,
}

impl<T: Deref<Target = [u8]>> IndexedBits<T> {
    /// Build the index for a sequence of bits.
    ///
    /// This is an expensive operation which will examine
    /// all of the data input.
    pub fn build_from_bits(bits: Bits<T>) -> Self {
        let index = {
            let bits_ref = BitsRef::from(&bits);
            let index = vec![0u64; index_raw::index_size_for(bits_ref)];
            let mut index = index.into_boxed_slice();
            index_raw::build_index_for(bits_ref, &mut index)
                .expect("Specifically made index of the right size");
            index
        };
        IndexedBits { index, bits }
    }

    /// Build an indexed bitvector from some bytes.
    ///
    /// This is the same as using `Bits::from_bytes`
    /// and `IndexedBits::build_from_bits`
    pub fn build_from_bytes(bytes: T, len: u64) -> Option<Self> {
        Bits::from_bytes(bytes, len).map(IndexedBits::build_from_bits)
    }

    fn index(&self) -> &[u64] {
        self.index.deref()
    }

    #[inline]
    pub fn bits(&self) -> Bits<&[u8]> {
        self.bits.clone_ref()
    }

    #[inline]
    fn bits_ref(&self) -> BitsRef {
        self.bits().into()
    }

    /// Discard the index and get the original bit sequence storage back.
    #[inline]
    pub fn decompose(self) -> Bits<T> {
        self.bits
    }

    /// The number of bits in the storage.
    #[inline]
    pub fn len(&self) -> u64 {
        self.bits.len()
    }

    /// Get the byte at a specific index.
    ///
    /// Returns `None` for out-of-bounds.
    ///
    /// ```
    /// use indexed_bitvec::IndexedBits;
    /// let bits = IndexedBits::build_from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.get(0), Some(true));
    /// assert_eq!(bits.get(7), Some(false));
    /// assert_eq!(bits.get(14), Some(true));
    /// assert_eq!(bits.get(15), None);
    /// ```
    #[inline]
    pub fn get(&self, idx_bits: u64) -> Option<bool> {
        self.bits().get(idx_bits)
    }

    /// Count the set bits (fast *O(1)*).
    ///
    /// ```
    /// use indexed_bitvec::IndexedBits;
    /// let bits = IndexedBits::build_from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.count_ones(), 14);
    /// assert_eq!(bits.count_zeros(), 1);
    /// assert_eq!(bits.count_ones() + bits.count_zeros(), bits.len());
    /// ```
    #[inline]
    pub fn count_ones(&self) -> u64 {
        index_raw::count_ones(self.index(), self.bits_ref())
    }

    /// Count the unset bits (fast *O(1)*).
    ///
    /// ```
    /// use indexed_bitvec::IndexedBits;
    /// let bits = IndexedBits::build_from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.count_ones(), 14);
    /// assert_eq!(bits.count_zeros(), 1);
    /// assert_eq!(bits.count_ones() + bits.count_zeros(), bits.len());
    /// ```
    #[inline]
    pub fn count_zeros(&self) -> u64 {
        index_raw::count_zeros(self.index(), self.bits_ref())
    }

    /// Count the set bits before a position in the bits (*O(1)*).
    ///
    /// Returns `None` it the index is out of bounds.
    ///
    /// ```
    /// use indexed_bitvec::IndexedBits;
    /// let bits = IndexedBits::build_from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert!((0..bits.len()).all(|idx|
    ///     bits.rank_ones(idx).unwrap()
    ///     + bits.rank_zeros(idx).unwrap()
    ///     == (idx as u64)));
    /// assert_eq!(bits.rank_ones(7), Some(7));
    /// assert_eq!(bits.rank_zeros(7), Some(0));
    /// assert_eq!(bits.rank_ones(8), Some(7));
    /// assert_eq!(bits.rank_zeros(8), Some(1));
    /// assert_eq!(bits.rank_ones(9), Some(8));
    /// assert_eq!(bits.rank_zeros(9), Some(1));
    /// assert_eq!(bits.rank_ones(15), None);
    /// ```
    #[inline]
    pub fn rank_ones(&self, idx: u64) -> Option<u64> {
        index_raw::rank_ones(self.index(), self.bits_ref(), idx)
    }

    /// Count the unset bits before a position in the bits (*O(1)*).
    ///
    /// Returns `None` it the index is out of bounds.
    ///
    /// ```
    /// use indexed_bitvec::IndexedBits;
    /// let bits = IndexedBits::build_from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert!((0..bits.len()).all(|idx|
    ///     bits.rank_ones(idx).unwrap()
    ///     + bits.rank_zeros(idx).unwrap()
    ///     == (idx as u64)));
    /// assert_eq!(bits.rank_ones(7), Some(7));
    /// assert_eq!(bits.rank_zeros(7), Some(0));
    /// assert_eq!(bits.rank_ones(8), Some(7));
    /// assert_eq!(bits.rank_zeros(8), Some(1));
    /// assert_eq!(bits.rank_ones(9), Some(8));
    /// assert_eq!(bits.rank_zeros(9), Some(1));
    /// assert_eq!(bits.rank_ones(15), None);
    /// ```
    #[inline]
    pub fn rank_zeros(&self, idx: u64) -> Option<u64> {
        index_raw::rank_zeros(self.index(), self.bits_ref(), idx)
    }

    /// Find the position of a set bit by its rank (*O(log n)*).
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank_ones(result) == Some(target_rank)`
    /// and `get(result) == Some(true)`.
    ///
    /// ```
    /// use indexed_bitvec::IndexedBits;
    /// let bits = IndexedBits::build_from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.select_ones(6), Some(6));
    /// assert_eq!(bits.select_ones(7), Some(8));
    /// assert_eq!(bits.select_zeros(0), Some(7));
    /// assert_eq!(bits.select_zeros(1), None);
    /// ```
    #[inline]
    pub fn select_ones(&self, target_rank: u64) -> Option<u64> {
        index_raw::select_ones(self.index(), self.bits_ref(), target_rank)
    }

    /// Find the position of an unset bit by its rank (*O(log n)*).
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank_zeros(result) == Some(target_rank)`
    /// and `get(result) == Some(false)`.
    ///
    /// ```
    /// use indexed_bitvec::IndexedBits;
    /// let bits = IndexedBits::build_from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.select_ones(6), Some(6));
    /// assert_eq!(bits.select_ones(7), Some(8));
    /// assert_eq!(bits.select_zeros(0), Some(7));
    /// assert_eq!(bits.select_zeros(1), None);
    #[inline]
    pub fn select_zeros(&self, target_rank: u64) -> Option<u64> {
        index_raw::select_zeros(self.index(), self.bits_ref(), target_rank)
    }
}

#[cfg(feature = "implement_heapsize")]
impl<T: core::ops::Deref<Target = [u8]> + heapsize::HeapSizeOf> heapsize::HeapSizeOf
    for IndexedBits<T>
{
    fn heap_size_of_children(&self) -> usize {
        self.index.heap_size_of_children() + self.bits.heap_size_of_children()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::gen_bits;
    use proptest::collection::SizeRange;
    use proptest::prelude::*;

    type Bitvec = IndexedBits<Vec<u8>>;

    prop_compose! {
        fn gen_indexed_bits_inner(byte_len: SizeRange)
            (bits in gen_bits(byte_len))
             -> Bitvec
        {
            IndexedBits::build_from_bits(bits)
        }
    }

    pub fn gen_indexed_bits(byte_len: impl Into<SizeRange>) -> impl Strategy<Value = Bitvec> {
        gen_indexed_bits_inner(byte_len.into())
    }

    proptest! {
        #[test]
        fn test_bits_refs_and_decompose(original_bits in gen_bits(0..=1024)) {
            let indexed = IndexedBits::build_from_bits(original_bits.clone());
            prop_assert_eq!(original_bits.clone_ref(), indexed.bits());
            prop_assert_eq!(BitsRef::from(&original_bits), indexed.bits_ref());
            prop_assert_eq!(original_bits, indexed.decompose());
        }
    }

    fn from_bytes_or_panic<T: Deref<Target = [u8]>>(bytes: T, len: u64) -> IndexedBits<T> {
        IndexedBits::build_from_bits(Bits::from_bytes(bytes, len).expect("invalid bytes in test"))
    }

    #[test]
    fn test_basic_get() {
        let example_data = vec![0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];
        let max_len = 8 * 8;
        for len in 0..=max_len {
            let bits = from_bytes_or_panic(example_data.clone(), len);
            for i in 0..len {
                assert_eq!(Some(i / 8 == i % 8), bits.get(i));
            }
            for i in len..=(max_len + 1) {
                assert_eq!(None, bits.get(i));
            }
        }

        let example_data = vec![0xff, 0xc0];
        let bits = from_bytes_or_panic(example_data.clone(), 10);
        for i in 0..10 {
            assert_eq!(bits.get(i), Some(true), "Differed at position {}", i)
        }
        for i in 10..16 {
            assert_eq!(bits.get(i), None, "Differed at position {}", i)
        }
    }

    #[test]
    fn test_count_examples() {
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| from_bytes_or_panic(bytes_a, len);
        assert_eq!(12, make(16).count_ones());
        assert_eq!(4, make(16).count_zeros());
        assert_eq!(12, make(15).count_ones());
        assert_eq!(3, make(15).count_zeros());
        assert_eq!(11, make(14).count_ones());
        assert_eq!(3, make(14).count_zeros());
        assert_eq!(11, make(13).count_ones());
        assert_eq!(2, make(13).count_zeros());
        assert_eq!(10, make(12).count_ones());
        assert_eq!(2, make(12).count_zeros());
        assert_eq!(10, make(11).count_ones());
        assert_eq!(1, make(11).count_zeros());
        assert_eq!(9, make(10).count_ones());
        assert_eq!(1, make(10).count_zeros());
        assert_eq!(9, make(9).count_ones());
        assert_eq!(0, make(9).count_zeros());
        assert_eq!(8, make(8).count_ones());
        assert_eq!(0, make(8).count_zeros());
        assert_eq!(7, make(7).count_ones());
        assert_eq!(0, make(7).count_zeros());
        assert_eq!(0, make(0).count_ones());
        assert_eq!(0, make(0).count_zeros());
    }

    fn test_count_via_get(bits: Bitvec, bit_to_count: bool) -> Result<(), TestCaseError> {
        fn inner<F>(bits: Bitvec, bit_to_count: bool, f: F) -> Result<(), TestCaseError>
        where
            F: Fn(&Bitvec) -> u64,
        {
            let count_via_get = (0..bits.len())
                .filter(|&idx| bits.get(idx).unwrap() == bit_to_count)
                .count() as u64;
            prop_assert_eq!(count_via_get, f(&bits));
            Ok(())
        }

        if bit_to_count {
            inner(bits, true, IndexedBits::count_ones)
        } else {
            inner(bits, false, IndexedBits::count_zeros)
        }
    }

    proptest! {
        #[test]
        fn test_count_ones_via_get(bits in gen_indexed_bits(0..=1024)) {
            test_count_via_get(bits, true)?;
        }

        #[test]
        fn test_count_ones_via_iter(bits in gen_indexed_bits(0..=1024)) {
            let count_via_iter =
                bits.bits().iter()
                .filter(|&b| b)
                .count() as u64;
            prop_assert_eq!(count_via_iter, bits.count_ones());

        }

        #[test]
        fn test_count_zeros_via_get(bits in gen_indexed_bits(0..=1024)) {
            test_count_via_get(bits, false)?;
        }

        #[test]
        fn test_count_zeros_via_count_ones(bits in gen_indexed_bits(0..=1024)) {
            prop_assert_eq!(bits.len() - bits.count_ones(), bits.count_zeros());
        }

    }

    #[test]
    fn test_rank_examples() {
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| from_bytes_or_panic(bytes_a, len);
        let bits_a = make(16);
        for i in 0..15 {
            assert_eq!(Some(make(i).count_ones()), bits_a.rank_ones(i));
            assert_eq!(Some(make(i).count_zeros()), bits_a.rank_zeros(i));
        }
        assert_eq!(None, bits_a.rank_ones(16));
        assert_eq!(None, bits_a.rank_zeros(16));
        assert_eq!(None, make(13).rank_ones(13));
        assert_eq!(None, make(13).rank_zeros(13));
        assert_eq!(bits_a.rank_ones(12), make(13).rank_ones(12));
        assert_eq!(bits_a.rank_zeros(12), make(13).rank_zeros(12));
    }

    fn test_rank_via_get(bits: Bitvec, bit_to_rank: bool) -> Result<(), TestCaseError> {
        fn inner<F>(bits: Bitvec, bit_to_rank: bool, f: F) -> Result<(), TestCaseError>
        where
            F: Fn(&Bitvec, u64) -> Option<u64>,
        {
            let mut running_rank = 0;
            for idx in 0..=(bits.len() + 64) {
                if idx >= bits.len() {
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
            inner(bits, true, IndexedBits::rank_ones)
        } else {
            inner(bits, false, IndexedBits::rank_zeros)
        }
    }

    proptest! {
        #[test]
        fn test_rank_ones_via_get(bits in gen_indexed_bits(0..=1024)) {
            test_rank_via_get(bits, true)?;
        }

        #[test]
        fn test_rank_ones_via_iter(bits in gen_indexed_bits(0..=1024)) {
            let mut idx = 0u64;
            let mut running_rank_ones = 0;
            for b in bits.bits().iter() {
                prop_assert_eq!(Some(running_rank_ones), bits.rank_ones(idx),
                                "disagree at index {}", idx);
                idx += 1;
                if b { running_rank_ones += 1 };
            }
            prop_assert_eq!(None, bits.rank_ones(idx),
                            "should be out of range at {}", idx);
        }

        #[test]
        fn test_rank_zeros_via_get(bits in gen_indexed_bits(0..=1024)) {
            test_rank_via_get(bits, false)?;
        }

        #[test]
        fn test_rank_zeros_via_rank_ones(bits in gen_indexed_bits(0..=1024)) {
            for idx in 0..=(bits.len() + 64) {
                let via_rank_ones =
                    bits.rank_ones(idx).map(|ones| idx - ones);
                prop_assert_eq!(via_rank_ones, bits.rank_zeros(idx));
            }
        }
    }

    #[test]
    fn test_select_examples() {
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| from_bytes_or_panic(bytes_a, len);
        assert_eq!(Some(14), make(16).select_ones(11));
        assert_eq!(None, make(14).select_ones(11));
    }

    fn test_select_via_count_rank_get(
        bits: Bitvec,
        bit_to_select: bool,
    ) -> Result<(), TestCaseError> {
        fn inner<C, R, S>(
            bits: Bitvec,
            bit_to_select: bool,
            count: C,
            rank: R,
            select: S,
        ) -> Result<(), TestCaseError>
        where
            C: Fn(&Bitvec) -> u64,
            R: Fn(&Bitvec, u64) -> Option<u64>,
            S: Fn(&Bitvec, u64) -> Option<u64>,
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
                IndexedBits::count_ones,
                IndexedBits::rank_ones,
                IndexedBits::select_ones,
            )
        } else {
            inner(
                bits,
                false,
                IndexedBits::count_zeros,
                IndexedBits::rank_zeros,
                IndexedBits::select_zeros,
            )
        }
    }

    proptest! {
        #[test]
        fn test_select_ones_via_count_rank_get(bits in gen_indexed_bits(0..=1024)) {
            test_select_via_count_rank_get(bits, true)?;
        }

        #[test]
        fn test_select_zeros_via_count_rank_get(bits in gen_indexed_bits(0..=1024)) {
            test_select_via_count_rank_get(bits, false)?;
        }
    }

    proptest! {
        #[test]
        fn test_serialise_roundtrip(original in gen_indexed_bits(0..=1024)) {
            let serialised = bincode::serialize(&original).unwrap();
            let deserialised: Bitvec = bincode::deserialize(&serialised).unwrap();
            prop_assert_eq!(original.bits(), deserialised.bits());
            prop_assert_eq!(original.index, deserialised.index);
        }
    }

    #[test]
    fn test_succinct_trie_bitvec() {
        // This bitvec was found to break some things that had previously been
        // believed to be invariants of the indexing - specifically the amount
        // of extra samples that might exist in the sampling index.
        let src_data = include_bytes!("../examples/strange-cases/succinct-trie.bin");
        let bitvec: Bits<Vec<u8>> = bincode::deserialize(src_data).unwrap();
        let bits = bitvec.clone_ref();
        assert_eq!(1178631, bits.len());
        assert_eq!(589316, bits.count_ones());
        assert_eq!(589315, bits.count_zeros());
        let bits = IndexedBits::build_from_bits(bits);

        let mut running_rank_ones = 0u64;
        let mut running_rank_zeros = 0u64;
        for (idx, bit) in bits.bits().iter().enumerate() {
            let idx = idx as u64;
            assert_eq!(Some(running_rank_ones), bits.rank_ones(idx));
            assert_eq!(Some(running_rank_zeros), bits.rank_zeros(idx));
            if bit {
                assert_eq!(idx, bits.select_ones(running_rank_ones).unwrap());
                running_rank_ones += 1;
            } else {
                assert_eq!(idx, bits.select_zeros(running_rank_zeros).unwrap());
                running_rank_zeros += 1;
            }
        }
    }
}

#[cfg(test)]
pub use self::tests::gen_indexed_bits;
