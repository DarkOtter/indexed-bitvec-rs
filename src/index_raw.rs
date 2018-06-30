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
//! This module contains the raw functions for the index rank
//! and select operations, as well as building indexes.
//!
//! The functions here do minimal if any checking on the size
//! or validity of indexes vs. the bitvectors they are used with,
//! so you may run into panics from e.g. out of bounds accesses
//! to slices. They should all be memory-safe though.

use super::{ceil_div, ceil_div_u64};
use std::cmp::min;
use bits_type::Bits;
use ones_or_zeros::{OneBits, ZeroBits, OnesOrZeros};

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

    pub const SAMPLE_LENGTH: u64 = 8192;

    /// e.g. if we have N one bits, how many samples do we need?
    pub fn samples_for_bits(matching_bitcount: u64) -> usize {
        ceil_div_u64(matching_bitcount, SAMPLE_LENGTH) as usize
    }
    /// If we have N one and zero bits,
    /// how many words for ones and zeros samples together?
    pub fn sample_words(total_bits: u64) -> usize {
        ceil_div(samples_for_bits(total_bits) + 1, 2)
    }

    pub fn total_index_words(total_bits: u64) -> usize {
        l0(total_bits) + l1l2(total_bits) + sample_words(total_bits)
    }

    pub const L1_BLOCKS_PER_L0_BLOCK: usize = (BITS_PER_L0_BLOCK / BITS_PER_L1_BLOCK) as usize;
    pub const L2_BLOCKS_PER_L1_BLOCK: usize = (BITS_PER_L1_BLOCK / BITS_PER_L2_BLOCK) as usize;

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
        pub fn pack(base_rank: u32, sub_ranks: [u16; 3]) -> Self {
            debug_assert!(sub_ranks.iter().all(|&x| (x & !0x03FF) == 0));
            L1L2Entry(
                ((base_rank as u64) << 32) | ((sub_ranks[0] as u64) << 22) |
                ((sub_ranks[1] as u64) << 12) | ((sub_ranks[2] as u64) << 2),
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

        pub fn sub_rank(self, i: usize) -> u64 {
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


    fn cast_to_l1l2<'a>(data: &'a [u64]) -> &'a [L1L2Entry] {
        use std::mem::{size_of, align_of};
        debug_assert_eq!(size_of::<u64>(), size_of::<L1L2Entry>());
        debug_assert_eq!(align_of::<u64>(), align_of::<L1L2Entry>());

        unsafe {
            use std::slice::from_raw_parts;
            let n = data.len();
            let ptr = data.as_ptr() as *mut L1L2Entry;
            from_raw_parts(ptr, n)
        }
    }

    fn cast_to_l1l2_mut<'a>(data: &'a mut [u64]) -> &'a mut [L1L2Entry] {
        use std::mem::{size_of, align_of};
        debug_assert_eq!(size_of::<u64>(), size_of::<L1L2Entry>());
        debug_assert_eq!(align_of::<u64>(), align_of::<L1L2Entry>());

        unsafe {
            use std::slice::from_raw_parts_mut;
            let n = data.len();
            let ptr = data.as_mut_ptr() as *mut L1L2Entry;
            from_raw_parts_mut(ptr, n)
        }
    }

    fn cast_to_samples<'a>(data: &'a [u64]) -> &'a [SampleEntry] {
        use std::mem::{size_of, align_of};
        debug_assert_eq!(size_of::<u64>(), 2 * size_of::<SampleEntry>());
        debug_assert_eq!(align_of::<u64>(), 2 * align_of::<SampleEntry>());

        unsafe {
            use std::slice::from_raw_parts;
            let n = data.len() * 2;
            let ptr = data.as_ptr() as *const SampleEntry;
            from_raw_parts(ptr, n)
        }
    }

    fn cast_to_samples_mut<'a>(data: &'a mut [u64]) -> &'a mut [SampleEntry] {
        use std::mem::{size_of, align_of};
        debug_assert_eq!(size_of::<u64>(), 2 * size_of::<SampleEntry>());
        debug_assert_eq!(align_of::<u64>(), 2 * align_of::<SampleEntry>());

        unsafe {
            use std::slice::from_raw_parts_mut;
            let n = data.len() * 2;
            let ptr = data.as_mut_ptr() as *mut SampleEntry;
            from_raw_parts_mut(ptr, n)
        }
    }


    pub fn split_l0<'a>(index: &'a [u64], data: Bits<&[u8]>) -> (&'a [u64], &'a [u64]) {
        index.split_at(size::l0(data.used_bits()))
    }

    pub fn split_l0_mut<'a>(index: &'a mut [u64], data: Bits<&[u8]>) -> (&'a mut [u64], &'a mut [u64]) {
        index.split_at_mut(size::l0(data.used_bits()))
    }

    pub fn split_l1l2<'a>(index_after_l0: &'a [u64], data: Bits<&[u8]>) -> (&'a [L1L2Entry], &'a [u64]) {
        let (l1l2, other) = index_after_l0.split_at(size::l1l2(data.used_bits()));
        (cast_to_l1l2(l1l2), other)
    }

    pub fn split_l1l2_mut<'a>(index_after_l0: &'a mut [u64], data: Bits<&[u8]>) -> (&'a mut [L1L2Entry], &'a mut [u64]) {
        let (l1l2, other) = index_after_l0.split_at_mut(size::l1l2(data.used_bits()));
        (cast_to_l1l2_mut(l1l2), other)
    }

    pub fn split_samples<'a>(index_after_l1l2: &'a [u64], data: Bits<&[u8]>, count_ones: u64) -> (&'a [SampleEntry], &'a [SampleEntry]) {
        let all_samples = cast_to_samples(index_after_l1l2);
        let n_samples_ones = size::samples_for_bits(count_ones);
        let n_samples_zeros = size::samples_for_bits(data.used_bits() - count_ones);
        let (ones_samples, other_samples) = all_samples.split_at(n_samples_ones);
        let zeros_samples = &other_samples[..n_samples_zeros];
        (ones_samples, zeros_samples)
    }

    pub fn split_samples_mut<'a>(index_after_l1l2: &'a mut [u64], data: Bits<&[u8]>, count_ones: u64) -> (&'a mut [SampleEntry], &'a mut [SampleEntry]) {
        let all_samples = cast_to_samples_mut(index_after_l1l2);
        let n_samples_ones = size::samples_for_bits(count_ones);
        let n_samples_zeros = size::samples_for_bits(data.used_bits() - count_ones);
        debug_assert!(all_samples.len() >= n_samples_ones + n_samples_zeros);
        debug_assert!(all_samples.len() <= n_samples_ones + n_samples_zeros + 1);
        let (ones_samples, other_samples) = all_samples.split_at_mut(n_samples_ones);
        let zeros_samples = &mut other_samples[..n_samples_zeros];
        (ones_samples, zeros_samples)
    }

    pub fn slice_l1l2_index(l1l2_index: &[L1L2Entry], l0_idx: usize) -> &[L1L2Entry] {
        let start_idx = l0_idx * size::L1_BLOCKS_PER_L0_BLOCK;
        let end_idx = min(start_idx + size::L1_BLOCKS_PER_L0_BLOCK, l1l2_index.len());
        &l1l2_index[start_idx..end_idx]
    }
}
use self::structure::{L1L2Entry, SampleEntry};

/// You need this many u64s for the index for these bits.
/// This calculation is O(1) (maths based on the number of bits).
pub fn index_size_for(bits: Bits<&[u8]>) -> usize {
    size::total_index_words(bits.used_bits())
}

use result::Error;

pub fn check_index_size(index: &[u64], bits: Bits<&[u8]>) -> Result<(), Error> {
    if index.len() != index_size_for(bits) {
        Err(Error::IndexIncorrectSize)
    } else {
        Ok(())
    }
}

pub fn build_index_for(bits: Bits<&[u8]>, into: &mut [u64]) -> Result<(), Error> {
    check_index_size(into, bits)?;

    if bits.used_bits() == 0 {
        debug_assert_eq!(0, into.len());
        return Ok(());
    }

    let (l0_index, index_after_l0) = structure::split_l0_mut(into, bits);
    let (l1l2_index, index_after_l1l2) = structure::split_l1l2_mut(index_after_l0, bits);

    // Build the L1L2 index, and get the L0 block bitcounts
    bits.chunks_bytes(size::BYTES_PER_L0_BLOCK)
        .zip(l1l2_index.chunks_mut(size::L1_BLOCKS_PER_L0_BLOCK))
        .zip(l0_index.iter_mut())
    // This loop could be parallelised
        .for_each(|((bits_chunk, l1l2_chunk), l0_entry)| {
            *l0_entry = build_inner_l1l2(l1l2_chunk, bits_chunk)
        });
    let l1l2_index: &[L1L2Entry] = l1l2_index;

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

/// Returns the total count of one bits
fn build_inner_l1l2(l1l2_index: &mut [L1L2Entry], data_chunk: Bits<&[u8]>) -> u64 {
    debug_assert!(data_chunk.used_bits() > 0);
    debug_assert!(data_chunk.used_bits() <= size::BITS_PER_L0_BLOCK);
    debug_assert!(l1l2_index.len() == size::l1l2(data_chunk.used_bits()));

    data_chunk.chunks_bytes(size::BYTES_PER_L1_BLOCK)
        .zip(l1l2_index.iter_mut())
    // This loop could be parallelised
        .for_each(|(l1_chunk, write_to)| {
            let mut counts = [0u16; 4];
            l1_chunk
                .chunks_bytes(size::BYTES_PER_L2_BLOCK)
                .zip(counts.iter_mut())
                .for_each(|(chunk, write_to)| {
                    *write_to = chunk.count::<OneBits>() as u16
                });

            counts[1] += counts[0];
            counts[2] += counts[1];
            counts[3] += counts[2];

            *write_to = L1L2Entry::pack(counts[3] as u32, [counts[0], counts[1], counts[2]]);
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

struct WithOffset<T> {
    offset: usize,
    data: T,
}

impl<T> WithOffset<T> {
    fn at_origin(data: T) -> Self {
        WithOffset { offset: 0, data }
    }

    fn offset_from_origin(&self) -> usize {
        self.offset
    }
}

impl<'a, T> WithOffset<&'a mut [T]> {
    fn split_at_mut_by_offset(self, idx: usize) -> (Self, Self) {
        let WithOffset { offset, data } = self;
        if idx < offset { panic!("Index out of bounds - before current offset") };
        let (data_l, data_r) = data.split_at_mut(idx - offset);
        let data_l_len = data_l.len();
        (WithOffset { offset, data: data_l },
         WithOffset { offset: offset + data_l_len, data: data_r })
    }
}

fn take_first_part_upto_offset<'a, T>(opt_ref: &mut Option<WithOffset<&'a mut [T]>>, idx: usize) -> Option<WithOffset<&'a mut [T]>> {
    let whole_thing = match opt_ref.take() {
        None => return None,
        Some(x) => x,
    };
    let (first_part, second_part) = whole_thing.split_at_mut_by_offset(idx);
    *opt_ref = Some(second_part);
    Some(first_part)
}

use std::ops::Deref;
impl<T> Deref for WithOffset<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

fn build_samples<W: OnesOrZeros>(l0_index: &[u64], l1l2_index: &[L1L2Entry], bits: Bits<&[u8]>, samples: &mut [SampleEntry]) {
    let mut running_base_rank = 0u64;
    let mut running_total_bits = 0u64;

    let l0_chunks_start_end_rank = {
        l0_index.iter().map(|cumulative_count| {
            let base_rank = running_base_rank.clone();
            running_total_bits = min(bits.used_bits(), running_total_bits + size::BITS_PER_L0_BLOCK);
            let cumulative_count = W::convert_count(cumulative_count.clone(), running_total_bits);
            running_base_rank = cumulative_count;
            (base_rank, cumulative_count)
        })
    };

    let chunks_with_samples =
        l0_chunks_start_end_rank
        .enumerate()
        .scan(Some(WithOffset::at_origin(samples)), |samples, (l0_idx, (rank_start, rank_end))| {
            let n_samples_seen_end = size::samples_for_bits(rank_end);
            match take_first_part_upto_offset(samples, n_samples_seen_end) {
                None => None,
                Some(here_samples) => {
                    if here_samples.len() == 0 {
                        None
                    } else {
                        debug_assert!(here_samples.len() == n_samples_seen_end - size::samples_for_bits(rank_start));
                        let inner_l1l2_index = structure::slice_l1l2_index(l1l2_index, l0_idx);
                        Some((rank_start, inner_l1l2_index, here_samples))
                    }
                },
            }
        });

    chunks_with_samples
        // This loop could be parallelised
        .for_each(|(start_rank, inner_l1l2_index, samples)| {
            build_samples_inner::<W>(start_rank, inner_l1l2_index, 0, inner_l1l2_index.len() * 4, samples)
        })
}

fn build_samples_inner<W: OnesOrZeros>(
    base_rank: u64,
    inner_l1l2_index: &[L1L2Entry],
    low_block: usize,
    high_block: usize,
    samples: WithOffset<&mut [SampleEntry]>) {

    if samples.len() == 0 {
        return;
    } else if samples.len() == 1 {
        debug_assert!(high_block > low_block);
        let target_rank = samples.offset_from_origin() as u64 * size::SAMPLE_LENGTH;
        let target_rank_in_l0 = target_rank - base_rank;
        let following_block_idx =
        binary_search(low_block, high_block, |block_idx| {
            read_l1l2_rank::<W>(inner_l1l2_index, block_idx) > target_rank_in_l0
        });
        debug_assert!(following_block_idx > low_block);
        samples.data[0] = SampleEntry::pack(following_block_idx - 1);
        return;
    }

    debug_assert!(samples.len() > 1);
    debug_assert!(high_block > low_block + 1);

    let mid_block = (low_block + high_block + 1) / 2;
    debug_assert!(mid_block < high_block);
    let samples_before_mid_block =
        size::samples_for_bits(read_l1l2_rank::<W>(inner_l1l2_index, mid_block) + base_rank);

    let (before_mid, after_mid) = samples.split_at_mut_by_offset(samples_before_mid_block);

    // These could be parallelised (divide-and-conquer)
    build_samples_inner::<W>(base_rank, inner_l1l2_index, low_block, mid_block, before_mid);
    build_samples_inner::<W>(base_rank, inner_l1l2_index, mid_block, high_block, after_mid);
}

pub fn count_ones(index: &[u64], bits: Bits<&[u8]>) -> u64 {
    if bits.used_bits() == 0 {
        return 0;
    }
    let l0_size = size::l0(bits.used_bits());
    debug_assert!(l0_size > 0);
    index[l0_size - 1]
}

pub fn count<W: OnesOrZeros>(index: &[u64], bits: Bits<&[u8]>) -> u64 {
    W::convert_count(count_ones(index, bits), bits.used_bits())
}

fn read_l0_cumulative_count<W: OnesOrZeros>(l0_index: &[u64], bits: Bits<&[u8]>, idx: usize) -> u64 {
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

fn read_l1l2_rank<W: OnesOrZeros>(inner_l1l2_index: &[L1L2Entry], block_idx: usize) -> u64 {
    let l1_idx = block_idx / size::L2_BLOCKS_PER_L1_BLOCK;
    let l2_idx = block_idx % size::L2_BLOCKS_PER_L1_BLOCK;
    let entry = inner_l1l2_index[l1_idx];
    let l1_rank_ones = entry.base_rank();
    let l2_rank_ones = if l2_idx > 0 { entry.sub_rank(l2_idx - 1) } else { 0 };
    W::convert_count(l1_rank_ones + l2_rank_ones, block_idx as u64 * size::BITS_PER_L2_BLOCK)
}

pub fn rank_ones(index: &[u64], bits: Bits<&[u8]>, idx: u64) -> Option<u64> {
    if idx >= bits.used_bits() {
        return None;
    } else if idx == 0 {
        return Some(0);
    }

    let (l0_index, index_after_l0) = structure::split_l0(index, bits);

    let l0_idx = idx / size::BITS_PER_L0_BLOCK;
    debug_assert!(l0_idx < l0_index.len() as u64);
    let l0_idx = l0_idx as usize;
    let l0_offset = idx % size::BITS_PER_L0_BLOCK;
    let l0_rank = read_l0_rank::<OneBits>(l0_index, bits, l0_idx);

    let (l1l2_index, _) = structure::split_l1l2(index_after_l0, bits);
    let inner_l1l2_index = structure::slice_l1l2_index(l1l2_index, l0_idx);

    let block_idx = l0_offset / size::BITS_PER_L2_BLOCK;
    debug_assert!(block_idx < (inner_l1l2_index.len() as u64) * size::L2_BLOCKS_PER_L1_BLOCK as u64);
    let block_idx = block_idx as usize;
    let block_offset = l0_offset % size::BITS_PER_L2_BLOCK;
    let block_rank = read_l1l2_rank::<OneBits>(inner_l1l2_index, block_idx);

    let scan_skip_bytes =
        l0_idx * size::BYTES_PER_L0_BLOCK +
        block_idx * size::BYTES_PER_L2_BLOCK;
    let scan_bits = bits.skip_bytes(scan_skip_bytes);
    let scanned_rank = scan_bits.rank::<OneBits>(block_offset).expect("Already checked size");
    Some(l0_rank + block_rank + scanned_rank)
}

pub fn rank<W: OnesOrZeros>(index: &[u64], bits: Bits<&[u8]>, idx: u64) -> Option<u64> {
    rank_ones(index, bits, idx).map(|res_ones| W::convert_count(res_ones, idx))
}

/// Assumes that for some i s.t. from <= i < until: check(j) <=> j >= i
/// This returns such i.
fn binary_search<F>(from: usize, until: usize, check: F) -> usize
where F: Fn(usize) -> bool
{
    let mut false_up_to = from;
    let mut true_from = until;

    while false_up_to < true_from {
        let mid_ish = (false_up_to + true_from) >> 1;
        if check(mid_ish) {
            true_from = mid_ish;
        } else {
            false_up_to = mid_ish + 1;
        }
    }

    return true_from;
}

pub fn select<W: OnesOrZeros>(index: &[u64], bits: Bits<&[u8]>, target_rank: u64) -> Option<u64> {
    if bits.used_bits() == 0 {
        return None;
    }
    let (l0_index, index_after_l0) = structure::split_l0(index, bits);
    debug_assert!(l0_index.len() > 0);
    let total_count_ones = l0_index[l0_index.len() - 1];
    let total_count = W::convert_count(total_count_ones, bits.used_bits());
    if target_rank >= total_count {
        return None;
    }

    // Find the right l0 block by binary search
    let l0_idx =
        binary_search(0, l0_index.len(), |idx| {
            read_l0_cumulative_count::<W>(l0_index, bits, idx) > target_rank
        });
    debug_assert!(l0_idx < l0_index.len());
    let next_l0_block_rank = read_l0_cumulative_count::<W>(l0_index, bits, l0_idx);
    debug_assert!(next_l0_block_rank > target_rank);
    let l0_block_rank = read_l0_rank::<W>(l0_index, bits, l0_idx);
    debug_assert!(l0_block_rank <= target_rank);
    let target_rank_in_l0_block = target_rank - l0_block_rank;

    // Unpack the other parts of the index
    let (l1l2_index, index_after_l1l2) = structure::split_l1l2(index_after_l0, bits);
    let inner_l1l2_index = structure::slice_l1l2_index(l1l2_index, l0_idx);
    debug_assert!(inner_l1l2_index.len() > 0);
    let (select_ones_samples, select_zeros_samples) =
        structure::split_samples(index_after_l1l2, bits, total_count_ones);
    let select_samples =
        if W::is_ones() { select_ones_samples } else { select_zeros_samples };

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
            inner_l1l2_index.len() * 4
        } else if next_sample_idx >= select_samples.len() as u64 {
            // Sample does not exist
            inner_l1l2_index.len() * 4
        } else {
            select_samples[next_sample_idx as usize].block_idx_in_l0_block()
        }
    };

    let block_idx = {
        let following_block_idx = 
            binary_search(block_idx_should_be_at_least,
                          block_idx_should_be_less_than,
                          |idx| {
                              read_l1l2_rank::<W>(inner_l1l2_index, idx) > target_rank_in_l0_block
                          });
        debug_assert!(following_block_idx > 0);
        following_block_idx - 1
    };
    let block_rank = read_l1l2_rank::<W>(inner_l1l2_index, block_idx);
    let target_rank_in_block = target_rank_in_l0_block - block_rank;

    let scan_skip_bytes =
        l0_idx * size::BYTES_PER_L0_BLOCK +
        block_idx * size::BYTES_PER_L2_BLOCK;
    let scan_bits = bits.skip_bytes(scan_skip_bytes);
    let scanned_idx =
        scan_bits.select::<W>(target_rank_in_block)
        .expect("Already checked against total count");

    Some(scan_skip_bytes as u64 * 8 + scanned_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ceil_div_u64;
    use quickcheck;
    use quickcheck::Arbitrary;
    use ones_or_zeros::{OneBits, ZeroBits};

    #[test]
    fn small_indexed_tests() {
        use rand::{Rng, SeedableRng, XorShiftRng};
        let n_bits: u64 = (1 << 19) - 1;
        let n_bytes: usize = ceil_div_u64(n_bits, 8) as usize;
        let seed = [42, 349107693, 17273721493, 135691827498];
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
            build_index_for(data, &mut index);
            index
        };

        let count_ones = data.count::<OneBits>();
        let count_zeros = n_bits - count_ones;
        assert_eq!(count_ones, count::<OneBits>(&index, data));
        assert_eq!(count_zeros, count::<ZeroBits>(&index, data));

        assert_eq!(None, rank::<OneBits>(&index, data, n_bits));
        assert_eq!(None, rank::<ZeroBits>(&index, data, n_bits));

        let rank_idxs = {
            let mut idxs: Vec<u64> = (0..1000).map(|_| rng.gen_range(0, n_bits)).collect();
            idxs.sort();
            idxs
        };
        for idx in rank_idxs {
            assert_eq!(data.rank::<OneBits>(idx), rank::<OneBits>(&index, data, idx));
            assert_eq!(data.rank::<ZeroBits>(idx), rank::<ZeroBits>(&index, data, idx));
        }

        assert_eq!(None, select::<OneBits>(&index, data, count_ones));
        let one_ranks = {
            let mut ranks: Vec<u64> = (0..1000).map(|_| rng.gen_range(0, count_ones)).collect();
            ranks.sort();
            ranks
        };
        for rank in one_ranks {
            assert_eq!(data.select::<OneBits>(rank), select::<OneBits>(&index, data, rank));
        }

        assert_eq!(None, select::<ZeroBits>(&index, data, count_zeros));
        let zero_ranks = {
            let mut ranks: Vec<u64> = (0..1000).map(|_| rng.gen_range(0, count_zeros)).collect();
            ranks.sort();
            ranks
        };
        for rank in zero_ranks {
            assert_eq!(data.select::<ZeroBits>(rank), select::<ZeroBits>(&index, data, rank));
        }
    }
}
