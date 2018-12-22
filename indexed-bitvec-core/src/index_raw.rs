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
//! The raw functions for building and using rank/select indexes.
//!
//! The functions here do minimal if any checking on the size
//! or validity of indexes vs. the bitvectors they are used with,
//! so you may run into panics from e.g. out of bounds accesses
//! to slices. They should all be memory-safe though.

use super::with_offset::WithOffset;
use super::{ceil_div, ceil_div_u64};
use super::Bits;
use crate::ones_or_zeros::{OneBits, OnesOrZeros, ZeroBits};
use core::cmp::min;

impl<'data> Bits<&'data [u8]> {
    /// Split the bits into a sequence of chunks of up to *n* bytes.
    fn chunks_by_bytes<'s>(&'s self, bytes_per_chunk: usize) -> impl Iterator<Item = Bits<&'s [u8]>> {

        let bits_per_chunk = (bytes_per_chunk as u64) * 8;
        self.bytes().chunks(bytes_per_chunk).enumerate().map(
            move |(i, chunk)| {
                let used_bits = i as u64 * bits_per_chunk;
                let bits = min(self.used_bits() - used_bits, bits_per_chunk);
                Bits::from(chunk, bits).expect("Size invariant violated")
            },
        )
    }

    /// Drop the first *n* bytes of bits from the front of the sequence.
    fn drop_bytes<'s>(&'s self, n_bytes: usize) -> Bits<&'s [u8]> {
        let bytes = self.bytes();
        if n_bytes >= bytes.len() {
            panic!("Index out of bounds: tried to drop all of the bits");
        }
        Bits::from(&bytes[n_bytes..], self.used_bits() - (n_bytes as u64 * 8))
            .expect("Checked sufficient bytes are present")
    }
}

mod size {
    use super::*;

    pub const BITS_PER_L0_BLOCK: u64 = 1 << 32;
    pub const BITS_PER_L1_BLOCK: u64 = BITS_PER_L2_BLOCK * 4;
    pub const BITS_PER_L2_BLOCK: u64 = 512;

    pub const BYTES_PER_L0_BLOCK: usize = (BITS_PER_L0_BLOCK / 8) as usize;
    pub const BYTES_PER_L1_BLOCK: usize = (BITS_PER_L1_BLOCK / 8) as usize;
    pub const BYTES_PER_L2_BLOCK: usize = (BITS_PER_L2_BLOCK / 8) as usize;

    pub fn l0(total_bits: u64) -> usize {
        ceil_div_u64(total_bits, BITS_PER_L0_BLOCK) as usize
    }

    pub fn l1l2(total_bits: u64) -> usize {
        ceil_div_u64(total_bits, BITS_PER_L1_BLOCK) as usize
    }

    pub fn blocks(total_bits: u64) -> usize {
        ceil_div_u64(total_bits, BITS_PER_L2_BLOCK) as usize
    }

    pub const SAMPLE_LENGTH: u64 = 8192;

    /// If we have *n* one bits (or zero bits), how many samples to cover those bits?
    pub fn samples_for_bits(matching_bitcount: u64) -> usize {
        ceil_div_u64(matching_bitcount, SAMPLE_LENGTH) as usize
    }
    /// If we have *n* one and zero bits, how many words for all samples together?
    pub fn sample_words(total_bits: u64) -> usize {
        ceil_div(samples_for_bits(total_bits) + 1, 2)
    }

    pub fn total_index_words(total_bits: u64) -> usize {
        l0(total_bits) + l1l2(total_bits) + sample_words(total_bits)
    }

    pub const L1_BLOCKS_PER_L0_BLOCK: usize = (BITS_PER_L0_BLOCK / BITS_PER_L1_BLOCK) as usize;
    pub const L2_BLOCKS_PER_L1_BLOCK: usize = (BITS_PER_L1_BLOCK / BITS_PER_L2_BLOCK) as usize;
    pub const L2_BLOCKS_PER_L0_BLOCK: usize = L2_BLOCKS_PER_L1_BLOCK * L1_BLOCKS_PER_L0_BLOCK;

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn bytes_evenly_divide_block_sizes() {
            assert_eq!(BITS_PER_L0_BLOCK % 8, 0);
            assert_eq!(BITS_PER_L1_BLOCK % 8, 0);
            assert_eq!(BITS_PER_L2_BLOCK % 8, 0);
        }

        #[test]
        fn l1l2_evenly_divide_l0() {
            // This property is needed so that the size of the l1l2
            // index works out correctly if calculated across separate
            // l0 blocks.
            assert_eq!(BITS_PER_L0_BLOCK % BITS_PER_L1_BLOCK, 0);
            assert_eq!(BITS_PER_L0_BLOCK % BITS_PER_L2_BLOCK, 0);
        }

        #[test]
        fn block_sizes_evenly_divide() {
            assert_eq!(BITS_PER_L0_BLOCK % BITS_PER_L1_BLOCK, 0);
            assert_eq!(BITS_PER_L1_BLOCK % BITS_PER_L2_BLOCK, 0);
        }

        #[test]
        fn sample_size_larger_than_l1() {
            // This is needed as we assume only one sample can be in each L1 block
            assert!(SAMPLE_LENGTH >= BITS_PER_L1_BLOCK);
        }
    }
}

mod structure {
    use super::*;

    #[derive(Copy, Clone, Debug)]
    pub struct L1L2Entry(u64);

    impl L1L2Entry {
        pub fn pack(base_rank: u32, first_counts: [u16; 3]) -> Self {
            debug_assert!(first_counts.iter().all(|&x| x < 0x0400));
            L1L2Entry(
                ((base_rank as u64) << 32) | ((first_counts[0] as u64) << 22) |
                    ((first_counts[1] as u64) << 12) | ((first_counts[2] as u64) << 2),
            )
        }

        pub fn base_rank(self) -> u64 {
            self.0 >> 32
        }

        fn fset_base_rank(self, base_rank: u32) -> Self {
            L1L2Entry(((base_rank as u64) << 32) | self.0 & 0xffffffff)
        }

        pub fn set_base_rank(&mut self, base_rank: u32) {
            *self = self.fset_base_rank(base_rank);
        }

        pub fn l2_count(self, i: usize) -> u64 {
            let shift = 22 - i * 10;
            (self.0 >> shift) & 0x3ff
        }
    }

    #[derive(Copy, Clone, Debug)]
    pub struct SampleEntry(u32);

    impl SampleEntry {
        pub fn pack(block_idx_in_l0_block: usize) -> Self {
            debug_assert!(block_idx_in_l0_block <= u32::max_value() as usize);
            SampleEntry(block_idx_in_l0_block as u32)
        }

        pub fn block_idx_in_l0_block(self) -> usize {
            self.0 as usize
        }
    }

    use core::mem::{align_of, size_of};

    fn cast_to_l1l2<'a>(data: &'a [u64]) -> &'a [L1L2Entry] {
        debug_assert_eq!(size_of::<u64>(), size_of::<L1L2Entry>());
        debug_assert_eq!(align_of::<u64>(), align_of::<L1L2Entry>());

        unsafe {
            use core::slice::from_raw_parts;
            let n = data.len();
            let ptr = data.as_ptr() as *mut L1L2Entry;
            from_raw_parts(ptr, n)
        }
    }

    fn cast_to_l1l2_mut<'a>(data: &'a mut [u64]) -> &'a mut [L1L2Entry] {
        debug_assert_eq!(size_of::<u64>(), size_of::<L1L2Entry>());
        debug_assert_eq!(align_of::<u64>(), align_of::<L1L2Entry>());

        unsafe {
            use core::slice::from_raw_parts_mut;
            let n = data.len();
            let ptr = data.as_mut_ptr() as *mut L1L2Entry;
            from_raw_parts_mut(ptr, n)
        }
    }

    fn cast_to_samples<'a>(data: &'a [u64]) -> &'a [SampleEntry] {
        debug_assert_eq!(size_of::<u64>(), 2 * size_of::<SampleEntry>());
        debug_assert_eq!(align_of::<u64>(), 2 * align_of::<SampleEntry>());

        unsafe {
            use core::slice::from_raw_parts;
            let n = data.len() * 2;
            let ptr = data.as_ptr() as *const SampleEntry;
            from_raw_parts(ptr, n)
        }
    }

    fn cast_to_samples_mut<'a>(data: &'a mut [u64]) -> &'a mut [SampleEntry] {
        debug_assert_eq!(size_of::<u64>(), 2 * size_of::<SampleEntry>());
        debug_assert_eq!(align_of::<u64>(), 2 * align_of::<SampleEntry>());

        unsafe {
            use core::slice::from_raw_parts_mut;
            let n = data.len() * 2;
            let ptr = data.as_mut_ptr() as *mut SampleEntry;
            from_raw_parts_mut(ptr, n)
        }
    }

    pub fn split_l0<'a>(index: &'a [u64], data: Bits<&[u8]>) -> (&'a [u64], &'a [u64]) {
        index.split_at(size::l0(data.used_bits()))
    }

    pub fn split_l0_mut<'a>(
        index: &'a mut [u64],
        data: Bits<&[u8]>,
    ) -> (&'a mut [u64], &'a mut [u64]) {
        index.split_at_mut(size::l0(data.used_bits()))
    }

    #[derive(Copy, Clone, Debug)]
    pub struct L1L2Indexes<'a>(&'a [L1L2Entry]);

    pub fn split_l1l2<'a>(
        index_after_l0: &'a [u64],
        data: Bits<&[u8]>,
    ) -> (L1L2Indexes<'a>, &'a [u64]) {
        let (l1l2, other) = index_after_l0.split_at(size::l1l2(data.used_bits()));
        (L1L2Indexes(cast_to_l1l2(l1l2)), other)
    }

    pub fn split_l1l2_mut<'a>(
        index_after_l0: &'a mut [u64],
        data: Bits<&[u8]>,
    ) -> (&'a mut [L1L2Entry], &'a mut [u64]) {
        let (l1l2, other) = index_after_l0.split_at_mut(size::l1l2(data.used_bits()));
        (cast_to_l1l2_mut(l1l2), other)
    }

    pub fn split_samples<'a>(
        index_after_l1l2: &'a [u64],
        data: Bits<&[u8]>,
        count_ones: u64,
    ) -> (&'a [SampleEntry], &'a [SampleEntry]) {
        let all_samples = cast_to_samples(index_after_l1l2);
        let n_samples_ones = size::samples_for_bits(count_ones);
        let n_samples_zeros = size::samples_for_bits(data.used_bits() - count_ones);
        let (ones_samples, other_samples) = all_samples.split_at(n_samples_ones);
        let zeros_samples = &other_samples[..n_samples_zeros];
        (ones_samples, zeros_samples)
    }

    pub fn split_samples_mut<'a>(
        index_after_l1l2: &'a mut [u64],
        data: Bits<&[u8]>,
        count_ones: u64,
    ) -> (&'a mut [SampleEntry], &'a mut [SampleEntry]) {
        let all_samples = cast_to_samples_mut(index_after_l1l2);
        let n_samples_ones = size::samples_for_bits(count_ones);
        let n_samples_zeros = size::samples_for_bits(data.used_bits() - count_ones);
        debug_assert!(all_samples.len() >= n_samples_ones + n_samples_zeros);
        debug_assert!(all_samples.len() <= n_samples_ones + n_samples_zeros + 1);
        let (ones_samples, other_samples) = all_samples.split_at_mut(n_samples_ones);
        let zeros_samples = &mut other_samples[..n_samples_zeros];
        (ones_samples, zeros_samples)
    }

    #[derive(Copy, Clone, Debug)]
    pub struct L1L2Index<'a> {
        block_count: usize,
        index_data: &'a [L1L2Entry],
    }

    impl<'a> L1L2Indexes<'a> {
        pub fn it_is_the_whole_index_honest(index: &'a [L1L2Entry]) -> Self {
            L1L2Indexes(index)
        }

        pub fn inner_index(self, all_bits: Bits<&[u8]>, l0_idx: usize) -> L1L2Index<'a> {
            let start_idx = l0_idx * size::L1_BLOCKS_PER_L0_BLOCK;
            let end_idx = min(start_idx + size::L1_BLOCKS_PER_L0_BLOCK, self.0.len());
            let block_count_to_end = size::blocks(all_bits.used_bits()) -
                start_idx * size::L2_BLOCKS_PER_L1_BLOCK;
            L1L2Index {
                block_count: min(block_count_to_end, size::L2_BLOCKS_PER_L0_BLOCK),
                index_data: &self.0[start_idx..end_idx],
            }
        }
    }


    impl<'a> L1L2Index<'a> {
        pub fn len(self) -> usize {
            self.block_count
        }

        pub fn rank_of_block<W: OnesOrZeros>(self, block_idx: usize) -> u64 {
            if block_idx >= self.block_count {
                panic!("Index out of bounds: not enough blocks");
            }

            let l1_idx = block_idx / size::L2_BLOCKS_PER_L1_BLOCK;
            let l2_idx = block_idx % size::L2_BLOCKS_PER_L1_BLOCK;
            let entry = self.index_data[l1_idx];
            let l1_rank_ones = entry.base_rank();
            let l2_rank_ones = {
                let mut l2_rank = 0;
                if l2_idx >= 3 {
                    l2_rank += entry.l2_count(2)
                }
                if l2_idx >= 2 {
                    l2_rank += entry.l2_count(1)
                }
                if l2_idx >= 1 {
                    l2_rank += entry.l2_count(0)
                }
                l2_rank
            };

            W::convert_count(
                l1_rank_ones + l2_rank_ones,
                block_idx as u64 * size::BITS_PER_L2_BLOCK,
            )
        }
    }
}
use self::structure::{L1L2Entry, SampleEntry, L1L2Index, L1L2Indexes};

/// Calculate the storage size for an index for a given bitvector (*O(1)*).
///
/// This just looks at the number of bits in the bitvector and does some
/// calculations. The number returned is the number of `u64`s needed to
/// store the index.
pub fn index_size_for(bits: Bits<&[u8]>) -> usize {
    size::total_index_words(bits.used_bits())
}

/// Indicates the index storage was the wrong size for the bit vector it was used with.
#[derive(Copy, Clone, Debug)]
pub struct IndexSizeError;

/// Check an index is the right size for a given bitvector.
///
/// This does not in any way guarantee the index was built for
/// that bitvector, or that neither has been modified.
pub fn check_index_size(index: &[u64], bits: Bits<&[u8]>) -> Result<(), IndexSizeError> {
    if index.len() != index_size_for(bits) {
        Err(IndexSizeError)
    } else {
        Ok(())
    }
}

/// Build the index data for a given bitvector (*O(n)*).
pub fn build_index_for(bits: Bits<&[u8]>, into: &mut [u64]) -> Result<(), IndexSizeError> {
    check_index_size(into, bits)?;

    if bits.used_bits() == 0 {
        debug_assert_eq!(0, into.len());
        return Ok(());
    }

    let (l0_index, index_after_l0) = structure::split_l0_mut(into, bits);
    let (l1l2_index, index_after_l1l2) = structure::split_l1l2_mut(index_after_l0, bits);

    // Build the L1L2 index, and get the L0 block bitcounts
    bits.chunks_by_bytes(size::BYTES_PER_L0_BLOCK)
        .zip(l1l2_index.chunks_mut(size::L1_BLOCKS_PER_L0_BLOCK))
        .zip(l0_index.iter_mut())
        .for_each(|((bits_chunk, l1l2_chunk), l0_entry)| {
            *l0_entry = build_inner_l1l2(l1l2_chunk, bits_chunk)
        });
    let l1l2_index = L1L2Indexes::it_is_the_whole_index_honest(l1l2_index);

    // Convert the L0 block bitcounts into the proper L0 index
    let mut total_count_ones = 0u64;
    for l0_entry in l0_index.iter_mut() {
        total_count_ones += l0_entry.clone();
        *l0_entry = total_count_ones;
    }
    let l0_index: &[u64] = l0_index;

    // Build the select index
    let (samples_ones, samples_zeros) =
        structure::split_samples_mut(index_after_l1l2, bits, total_count_ones);
    build_samples::<OneBits>(l0_index, l1l2_index, bits, samples_ones);
    build_samples::<ZeroBits>(l0_index, l1l2_index, bits, samples_zeros);

    Ok(())
}

/// Build the inner l1l2 index and return the total count of set bits.
fn build_inner_l1l2(l1l2_index: &mut [L1L2Entry], data_chunk: Bits<&[u8]>) -> u64 {
    debug_assert!(data_chunk.used_bits() > 0);
    debug_assert!(data_chunk.used_bits() <= size::BITS_PER_L0_BLOCK);
    debug_assert!(l1l2_index.len() == size::l1l2(data_chunk.used_bits()));

    data_chunk
        .chunks_by_bytes(size::BYTES_PER_L1_BLOCK)
        .zip(l1l2_index.iter_mut())
        .for_each(|(l1_chunk, write_to)| {
            let mut counts = [0u16; 3];
            let mut chunks = l1_chunk.chunks_by_bytes(size::BYTES_PER_L2_BLOCK);
            let count_or_zero =
                |opt: Option<Bits<&[u8]>>| opt.map_or(0, |chunk| chunk.count_ones() as u16);

            counts[0] = count_or_zero(chunks.next());
            counts[1] = count_or_zero(chunks.next());
            counts[2] = count_or_zero(chunks.next());
            let mut total = count_or_zero(chunks.next());
            total += counts[0];
            total += counts[1];
            total += counts[2];

            *write_to = L1L2Entry::pack(total as u32, counts);
        });

    // Pass through reassigning each entry to hold its rank to finish.
    let mut running_total = 0u64;
    for entry in l1l2_index.iter_mut() {
        let base_rank = running_total.clone() as u32;
        running_total += entry.base_rank();
        entry.set_base_rank(base_rank);
    }

    running_total
}

fn build_samples<W: OnesOrZeros>(
    l0_index: &[u64],
    l1l2_index: L1L2Indexes,
    all_bits: Bits<&[u8]>,
    samples: &mut [SampleEntry],
) {
    build_samples_outer::<W>(
        l0_index,
        0,
        l0_index.len(),
        l1l2_index,
        all_bits,
        WithOffset::at_origin(samples),
    )
}

fn build_samples_outer<W: OnesOrZeros>(
    l0_index: &[u64],
    low_l0_block: usize,
    high_l0_block: usize,
    l1l2_index: L1L2Indexes,
    all_bits: Bits<&[u8]>,
    samples: WithOffset<&mut [SampleEntry]>,
) {
    if low_l0_block >= high_l0_block || samples.len() == 0 {
        return;
    } else if low_l0_block + 1 >= high_l0_block {
        let l0_idx = low_l0_block;
        let base_rank = read_l0_rank::<W>(l0_index, all_bits, l0_idx);
        let inner_l1l2_index = l1l2_index.inner_index(all_bits, l0_idx);
        return build_samples_inner::<W>(
            base_rank,
            inner_l1l2_index,
            0,
            inner_l1l2_index.len(),
            samples,
        );
    }

    debug_assert!(low_l0_block + 1 < high_l0_block);
    let mid_l0_block = (low_l0_block + high_l0_block) / 2;
    debug_assert!(mid_l0_block > low_l0_block);
    debug_assert!(mid_l0_block < high_l0_block);

    let samples_before_mid_l0_block =
        size::samples_for_bits(read_l0_rank::<W>(l0_index, all_bits, mid_l0_block));
    let (before_mid, after_mid) = samples.split_at_mut_from_origin(samples_before_mid_l0_block);

    build_samples_outer::<W>(
        l0_index,
        low_l0_block,
        mid_l0_block,
        l1l2_index,
        all_bits,
        before_mid,
    );
    build_samples_outer::<W>(
        l0_index,
        mid_l0_block,
        high_l0_block,
        l1l2_index,
        all_bits,
        after_mid,
    );
}

fn build_samples_inner<W: OnesOrZeros>(
    base_rank: u64,
    inner_l1l2_index: L1L2Index,
    low_block: usize,
    high_block: usize,
    samples: WithOffset<&mut [SampleEntry]>,
) {
    if samples.len() == 0 {
        return;
    } else if samples.len() == 1 {
        debug_assert!(high_block > low_block);
        let target_rank = samples.offset_from_origin() as u64 * size::SAMPLE_LENGTH;
        let target_rank_in_l0 = target_rank - base_rank;
        let following_block_idx = binary_search(low_block, high_block, |block_idx| {
            inner_l1l2_index.rank_of_block::<W>(block_idx) > target_rank_in_l0
        });
        debug_assert!(following_block_idx > low_block);
        samples.decompose()[0] = SampleEntry::pack(following_block_idx - 1);
        return;
    }

    debug_assert!(samples.len() > 1);
    debug_assert!(low_block + 1 < high_block);
    let mid_block = (low_block + high_block) / 2;
    debug_assert!(mid_block > low_block);
    debug_assert!(mid_block < high_block);

    let samples_before_mid_block =
        size::samples_for_bits(inner_l1l2_index.rank_of_block::<W>(mid_block) + base_rank);

    let (before_mid, after_mid) = samples.split_at_mut_from_origin(samples_before_mid_block);

    build_samples_inner::<W>(
        base_rank,
        inner_l1l2_index,
        low_block,
        mid_block,
        before_mid,
    );
    build_samples_inner::<W>(
        base_rank,
        inner_l1l2_index,
        mid_block,
        high_block,
        after_mid,
    );
}

/// Count the set bits using the index (fast *O(1)*).
#[inline]
pub fn count_ones(index: &[u64], bits: Bits<&[u8]>) -> u64 {
    if bits.used_bits() == 0 {
        return 0;
    }
    let l0_size = size::l0(bits.used_bits());
    debug_assert!(l0_size > 0);
    index[l0_size - 1]
}

/// Count the unset bits using the index (fast *O(1)*).
#[inline]
pub fn count_zeros(index: &[u64], bits: Bits<&[u8]>) -> u64 {
    ZeroBits::convert_count(count_ones(index, bits), bits.used_bits())
}

fn read_l0_cumulative_count<W: OnesOrZeros>(
    l0_index: &[u64],
    bits: Bits<&[u8]>,
    idx: usize,
) -> u64 {
    let count_ones = l0_index[idx];
    let total_count = if idx + 1 < l0_index.len() {
        (idx as u64 + 1) * size::BITS_PER_L0_BLOCK
    } else {
        bits.used_bits()
    };
    W::convert_count(count_ones, total_count)
}

fn read_l0_rank<W: OnesOrZeros>(l0_index: &[u64], bits: Bits<&[u8]>, idx: usize) -> u64 {
    if idx > 0 {
        read_l0_cumulative_count::<W>(l0_index, bits, idx - 1)
    } else {
        0
    }
}

/// Count the set bits before a position in the bits using the index (*O(1)*).
///
/// Returns `None` it the index is out of bounds.
pub fn rank_ones(index: &[u64], all_bits: Bits<&[u8]>, idx: u64) -> Option<u64> {
    if idx >= all_bits.used_bits() {
        return None;
    } else if idx == 0 {
        return Some(0);
    }

    let (l0_index, index_after_l0) = structure::split_l0(index, all_bits);

    let l0_idx = idx / size::BITS_PER_L0_BLOCK;
    debug_assert!(l0_idx < l0_index.len() as u64);
    let l0_idx = l0_idx as usize;
    let l0_offset = idx % size::BITS_PER_L0_BLOCK;
    let l0_rank = read_l0_rank::<OneBits>(l0_index, all_bits, l0_idx);

    let (l1l2_index, _) = structure::split_l1l2(index_after_l0, all_bits);
    let inner_l1l2_index = l1l2_index.inner_index(all_bits, l0_idx);

    let block_idx = l0_offset / size::BITS_PER_L2_BLOCK;
    debug_assert!(
        block_idx < (inner_l1l2_index.len() as u64) * size::L2_BLOCKS_PER_L1_BLOCK as u64
    );
    let block_idx = block_idx as usize;
    let block_offset = l0_offset % size::BITS_PER_L2_BLOCK;
    let block_rank = inner_l1l2_index.rank_of_block::<OneBits>(block_idx);

    let scan_skip_bytes = l0_idx * size::BYTES_PER_L0_BLOCK + block_idx * size::BYTES_PER_L2_BLOCK;
    let scan_bits = all_bits.drop_bytes(scan_skip_bytes);
    let scanned_rank = scan_bits.rank_ones(block_offset).expect(
        "Already checked size",
    );
    Some(l0_rank + block_rank + scanned_rank)
}


/// Count the unset bits before a position in the bits using the index (*O(1)*).
///
/// Returns `None` it the index is out of bounds.
#[inline]
pub fn rank_zeros(index: &[u64], bits: Bits<&[u8]>, idx: u64) -> Option<u64> {
    rank_ones(index, bits, idx).map(|res_ones| ZeroBits::convert_count(res_ones, idx))
}

/// Find the index *i* which partitions the input space into values
/// satisfying the check and those which don't.
///
/// This assumes there is some *i* which is at least `from` and less
/// than `until` such that `check(j) == (j >= i)`.
fn binary_search<F>(from: usize, until: usize, check: F) -> usize
where
    F: Fn(usize) -> bool,
{
    const LINEAR_FOR_N: usize = 16;

    let mut false_up_to = from;
    let mut true_from = until;

    while false_up_to + LINEAR_FOR_N < true_from {
        let mid_ish = (false_up_to + true_from) / 2;
        if check(mid_ish) {
            true_from = mid_ish;
        } else {
            false_up_to = mid_ish + 1;
        }
    }

    while false_up_to < true_from && !check(false_up_to) {
        false_up_to += 1;
    }
    debug_assert!(false_up_to <= true_from);
    debug_assert!(false_up_to == true_from || check(false_up_to));

    return false_up_to;
}

fn select<W: OnesOrZeros>(index: &[u64], all_bits: Bits<&[u8]>, target_rank: u64) -> Option<u64> {
    if all_bits.used_bits() == 0 {
        return None;
    }
    let (l0_index, index_after_l0) = structure::split_l0(index, all_bits);
    debug_assert!(l0_index.len() > 0);
    let total_count_ones = l0_index[l0_index.len() - 1];
    let total_count = W::convert_count(total_count_ones, all_bits.used_bits());
    if target_rank >= total_count {
        return None;
    }

    // Find the right l0 block by binary search
    let l0_idx = binary_search(0, l0_index.len(), |idx| {
        read_l0_cumulative_count::<W>(l0_index, all_bits, idx) > target_rank
    });
    debug_assert!(l0_idx < l0_index.len());
    let next_l0_block_rank = read_l0_cumulative_count::<W>(l0_index, all_bits, l0_idx);
    debug_assert!(next_l0_block_rank > target_rank);
    let l0_block_rank = read_l0_rank::<W>(l0_index, all_bits, l0_idx);
    debug_assert!(l0_block_rank <= target_rank);
    let target_rank_in_l0_block = target_rank - l0_block_rank;

    // Unpack the other parts of the index
    let (l1l2_index, index_after_l1l2) = structure::split_l1l2(index_after_l0, all_bits);
    let inner_l1l2_index = l1l2_index.inner_index(all_bits, l0_idx);
    debug_assert!(inner_l1l2_index.len() > 0);
    let (select_ones_samples, select_zeros_samples) =
        structure::split_samples(index_after_l1l2, all_bits, total_count_ones);
    let select_samples = if W::is_ones() {
        select_ones_samples
    } else {
        select_zeros_samples
    };

    // Use the samples to find bounds on which block can contain our target bit
    let sample_idx = target_rank / size::SAMPLE_LENGTH;
    let block_idx_should_be_at_least = {
        let sample_rank = sample_idx * size::SAMPLE_LENGTH;
        if sample_rank < l0_block_rank {
            // Sample is from the previous l0 block
            0
        } else {
            select_samples[sample_idx as usize].block_idx_in_l0_block()
        }
    };
    let block_idx_should_be_less_than = {
        let next_sample_idx = sample_idx + 1;
        let next_sample_rank = next_sample_idx * size::SAMPLE_LENGTH;
        if next_sample_rank >= next_l0_block_rank {
            // Sample is in the next l0 block
            inner_l1l2_index.len()
        } else if next_sample_idx >= select_samples.len() as u64 {
            // Sample does not exist
            inner_l1l2_index.len()
        } else {
            select_samples[next_sample_idx as usize].block_idx_in_l0_block() + 1
        }
    };

    let block_idx = {
        let following_block_idx =
            binary_search(
                block_idx_should_be_at_least,
                block_idx_should_be_less_than,
                |idx| inner_l1l2_index.rank_of_block::<W>(idx) > target_rank_in_l0_block,
            );
        debug_assert!(following_block_idx > 0);
        following_block_idx - 1
    };
    let block_rank = inner_l1l2_index.rank_of_block::<W>(block_idx);
    let target_rank_in_block = target_rank_in_l0_block - block_rank;

    let scan_skip_bytes = l0_idx * size::BYTES_PER_L0_BLOCK + block_idx * size::BYTES_PER_L2_BLOCK;
    let scan_bits = all_bits.drop_bytes(scan_skip_bytes);
    let scanned_idx = scan_bits.select::<W>(target_rank_in_block).expect(
        "Already checked against total count",
    );

    Some(scan_skip_bytes as u64 * 8 + scanned_idx)
}

/// Find the position of a set bit by its rank using the index (*O(log n)*).
///
/// Returns `None` if no suitable bit is found. It is
/// always the case otherwise that `rank_ones(index, result) == Some(target_rank)`
/// and `get(result) == Some(true)`.
pub fn select_ones(index: &[u64], all_bits: Bits<&[u8]>, target_rank: u64) -> Option<u64> {
    select::<OneBits>(index, all_bits, target_rank)
}

/// Find the position of an unset bit by its rank using the index (*O(log n)*).
///
/// Returns `None` if no suitable bit is found. It is
/// always the case otherwise that `rank_zeros(index, result) == Some(target_rank)`
/// and `get(result) == Some(false)`.
pub fn select_zeros(index: &[u64], all_bits: Bits<&[u8]>, target_rank: u64) -> Option<u64> {
    select::<ZeroBits>(index, all_bits, target_rank)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec::Vec;

    #[test]
    fn select_bug_issue_15() {
        // When the bit we are selecting is in the same block as the next index sample
        let mut data = vec![0xffu8; 8192 / 8 * 2];
        data[8192 / 8 - 1] = 0;
        let data = Bits::from(data, 8192 * 2).unwrap();
        let data = data.clone_ref();
        let mut index = vec![0u64; index_size_for(data)];
        build_index_for(data, &mut index).unwrap();
        let index = index;
        assert_eq!(select_ones(&index, data, 8191), Some(8199));
    }

    #[test]
    fn small_indexed_tests() {
        use rand::{SeedableRng, XorShiftRng, RngCore, Rng};
        let n_bits: u64 = (1 << 19) - 1;
        let n_bytes: usize = ceil_div_u64(n_bits, 8) as usize;
        let seed = [
            42,
            73,
            197,
            231,
            255,
            43,
            87,
            05,
            50,
            13,
            74,
            107,
            195,
            231,
            5,
            1,
        ];
        let mut rng = XorShiftRng::from_seed(seed);
        let data = {
            let mut data = vec![0u8; n_bytes];
            rng.fill_bytes(&mut data);
            data
        };
        let data = Bits::from(data, n_bits).expect("Should have enough bytes");
        let data = data.clone_ref();
        let index = {
            let mut index = vec![0u64; index_size_for(data)];
            build_index_for(data, &mut index).unwrap();
            index
        };

        let expected_count_ones = data.count_ones();
        let expected_count_zeros = n_bits - expected_count_ones;
        assert_eq!(expected_count_ones, count_ones(&index, data));
        assert_eq!(expected_count_zeros, count_zeros(&index, data));

        assert_eq!(None, rank_ones(&index, data, n_bits));
        assert_eq!(None, rank_zeros(&index, data, n_bits));

        let rank_idxs = {
            let mut idxs: Vec<u64> = (0..1000).map(|_| rng.gen_range(0, n_bits)).collect();
            idxs.sort();
            idxs
        };
        for idx in rank_idxs {
            assert_eq!(data.rank_ones(idx), rank_ones(&index, data, idx));
            assert_eq!(data.rank_zeros(idx), rank_zeros(&index, data, idx));
        }

        assert_eq!(None, select_ones(&index, data, expected_count_ones));
        let one_ranks = {
            let mut ranks: Vec<u64> = (0..1000)
                .map(|_| rng.gen_range(0, expected_count_ones))
                .collect();
            ranks.sort();
            ranks
        };
        for rank in one_ranks {
            assert_eq!(data.select_ones(rank), select_ones(&index, data, rank));
        }

        assert_eq!(None, select_zeros(&index, data, expected_count_zeros));
        let zero_ranks = {
            let mut ranks: Vec<u64> = (0..1000)
                .map(|_| rng.gen_range(0, expected_count_zeros))
                .collect();
            ranks.sort();
            ranks
        };
        for rank in zero_ranks {
            assert_eq!(data.select_zeros(rank), select_zeros(&index, data, rank));
        }
    }
}
