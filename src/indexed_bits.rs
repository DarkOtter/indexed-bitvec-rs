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
use std::ops::{Deref, DerefMut};
use indexed_bitvec_core::bits::Bits;
use indexed_bitvec_core::index_raw;

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
            let bits_as_u8 = bits.clone_ref();
            let index = vec![0u64; index_raw::index_size_for(bits_as_u8)];
            let mut index = index.into_boxed_slice();
            index_raw::build_index_for(bits_as_u8, index.deref_mut())
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

    /// Discard the index and get the original bit sequence storage back.
    #[inline]
    pub fn decompose(self) -> Bits<T> {
        self.bits
    }

    /// Count the set bits (fast *O(1)*).
    #[inline]
    pub fn count_ones(&self) -> u64 {
        index_raw::count_ones(self.index(), self.bits())
    }

    /// Count the unset bits (fast *O(1)*).
    #[inline]
    pub fn count_zeros(&self) -> u64 {
        index_raw::count_zeros(self.index(), self.bits())
    }

    /// Count the set bits before a position in the bits (*O(1)*).
    ///
    /// Returns `None` it the index is out of bounds.
    #[inline]
    pub fn rank_ones(&self, idx: u64) -> Option<u64> {
        index_raw::rank_ones(self.index(), self.bits(), idx)
    }

    /// Count the unset bits before a position in the bits (*O(1)*).
    ///
    /// Returns `None` it the index is out of bounds.
    #[inline]
    pub fn rank_zeros(&self, idx: u64) -> Option<u64> {
        index_raw::rank_zeros(self.index(), self.bits(), idx)
    }

    /// Find the position of a set bit by its rank (*O(log n)*).
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank_ones(result) == Some(target_rank)`
    /// and `get(result) == Some(true)`.
    #[inline]
    pub fn select_ones(&self, target_rank: u64) -> Option<u64> {
        index_raw::select_ones(self.index(), self.bits(), target_rank)
    }

    /// Find the position of an unset bit by its rank (*O(log n)*).
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank_zeros(result) == Some(target_rank)`
    /// and `get(result) == Some(false)`.
    #[inline]
    pub fn select_zeros(&self, target_rank: u64) -> Option<u64> {
        index_raw::select_zeros(self.index(), self.bits(), target_rank)
    }
}

#[cfg(feature = "implement_heapsize")]
impl<T: core::ops::Deref<Target = [u8]> + heapsize::HeapSizeOf> heapsize::HeapSizeOf
    for IndexedBits<T> {
    fn heap_size_of_children(&self) -> usize {
        self.index.heap_size_of_children() + self.bits.heap_size_of_children()
    }
}
