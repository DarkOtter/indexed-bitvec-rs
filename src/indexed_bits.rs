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
    pub fn build_index(bits: Bits<T>) -> Self {
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

    /// Count the set bits (fast *O(1)*).
    #[inline]
    pub fn count_ones(&self) -> u64 {
        index_raw::count_ones(self.index(), self.bits_ref())
    }

    /// Count the unset bits (fast *O(1)*).
    #[inline]
    pub fn count_zeros(&self) -> u64 {
        index_raw::count_zeros(self.index(), self.bits_ref())
    }

    /// Count the set bits before a position in the bits (*O(1)*).
    ///
    /// Returns `None` it the index is out of bounds.
    #[inline]
    pub fn rank_ones(&self, idx: u64) -> Option<u64> {
        index_raw::rank_ones(self.index(), self.bits_ref(), idx)
    }

    /// Count the unset bits before a position in the bits (*O(1)*).
    ///
    /// Returns `None` it the index is out of bounds.
    #[inline]
    pub fn rank_zeros(&self, idx: u64) -> Option<u64> {
        index_raw::rank_zeros(self.index(), self.bits_ref(), idx)
    }

    /// Find the position of a set bit by its rank (*O(log n)*).
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank_ones(result) == Some(target_rank)`
    /// and `get(result) == Some(true)`.
    #[inline]
    pub fn select_ones(&self, target_rank: u64) -> Option<u64> {
        index_raw::select_ones(self.index(), self.bits_ref(), target_rank)
    }

    /// Find the position of an unset bit by its rank (*O(log n)*).
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank_zeros(result) == Some(target_rank)`
    /// and `get(result) == Some(false)`.
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

    prop_compose! {
        fn gen_indexed_bits_inner(byte_len: SizeRange)
            (bits in gen_bits(byte_len))
             -> IndexedBits<Vec<u8>>
        {
            IndexedBits::build_index(bits)
        }
    }

    pub fn gen_indexed_bits(
        byte_len: impl Into<SizeRange>,
    ) -> impl Strategy<Value = IndexedBits<Vec<u8>>> {
        gen_indexed_bits_inner(byte_len.into())
    }

    proptest! {
        #[test]
        fn test_bits_refs_and_decompose(original_bits in gen_bits(0..=1024)) {
            let indexed = IndexedBits::build_index(original_bits.clone());
            prop_assert_eq!(original_bits.clone_ref(), indexed.bits());
            prop_assert_eq!(BitsRef::from(&original_bits), indexed.bits_ref());
            prop_assert_eq!(original_bits, indexed.decompose());
        }
    }

    fn from_bytes_or_panic<T: Deref<Target = [u8]>>(bytes: T, len: u64) -> IndexedBits<T> {
        IndexedBits::build_index(Bits::from_bytes(bytes, len).expect("invalid bytes in test"))
    }

    // TODO: Test index bits
    // TODO: Test serialisation

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
        let bits = IndexedBits::build_index(bits);

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
