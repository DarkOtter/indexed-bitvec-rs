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
        (matching_bitcount.saturating_sub(1) / SAMPLE_LENGTH) as usize
    }
    /// If we have N one and zero bits,
    /// how many words for ones and zeros samples together?
    pub fn sample_words(total_bits: u64) -> usize {
        ceil_div(samples_for_bits(total_bits), 2)
    }

    pub fn total_index_words(total_bits: u64) -> usize {
        l0(total_bits) + l1l2(total_bits) + sample_words(total_bits)
    }

    pub const L1_BLOCKS_PER_L0_BLOCK: usize = (BITS_PER_L0_BLOCK / BITS_PER_L1_BLOCK) as usize;

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
        fn sample_size_larger_than_l1() {
            // This is needed as we assume only one sample can be in each L1 block
            assert!(SAMPLE_LENGTH >= BITS_PER_L1_BLOCK);
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct L1L2Entry(u64);

impl L1L2Entry {
    fn pack(base_rank: u32, sub_ranks: [u16; 3]) -> Self {
        L1L2Entry(
            ((base_rank as u64) << 32) | ((sub_ranks[0] as u64) << 22) |
                ((sub_ranks[1] as u64) << 12) | ((sub_ranks[2] as u64) << 2),
        )
    }

    fn base_rank(self) -> u64 {
        self.0 >> 32
    }

    fn fset_base_rank(self, base_rank: u32) -> Self {
        L1L2Entry(((base_rank as u64) << 32) | self.0 & 0xffffffff)
    }

    fn set_base_rank(&mut self, base_rank: u32) {
        *self = self.fset_base_rank(base_rank);
    }

    fn sub_rank(self, i: usize) -> u64 {
        let shift = 22 - i * 10;
        (self.0 >> shift) & 0x3ff
    }
}

impl From<u64> for L1L2Entry {
    fn from(i: u64) -> Self {
        L1L2Entry(i)
    }
}

impl From<L1L2Entry> for u64 {
    fn from(entry: L1L2Entry) -> Self {
        entry.0
    }
}

mod unpack {
    use super::*;

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

    fn cast_to_u32<'a>(data: &'a [u64]) -> &'a [u32] {
        use std::mem::{size_of, align_of};
        debug_assert_eq!(size_of::<u64>(), 2 * size_of::<u32>());
        debug_assert_eq!(align_of::<u64>(), 2 * align_of::<u32>());

        unsafe {
            use std::slice::from_raw_parts;
            let n = data.len() * 2;
            let ptr = data.as_ptr() as *const u32;
            from_raw_parts(ptr, n)
        }
    }

    fn cast_to_u32_mut<'a>(data: &'a mut [u64]) -> &'a mut [u32] {
        use std::mem::{size_of, align_of};
        debug_assert_eq!(size_of::<u64>(), 2 * size_of::<u32>());
        debug_assert_eq!(align_of::<u64>(), 2 * align_of::<u32>());

        unsafe {
            use std::slice::from_raw_parts_mut;
            let n = data.len() * 2;
            let ptr = data.as_mut_ptr() as *mut u32;
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

    pub fn split_samples<'a>(index_after_l1l2: &'a [u64], data: Bits<&[u8]>, count_ones: u64) -> (&'a [u32], &'a [u32]) {
        let all_samples = cast_to_u32(index_after_l1l2);
        let n_samples_ones = size::samples_for_bits(count_ones);
        let n_samples_zeros = size::samples_for_bits(data.used_bits() - count_ones);
        let (ones_samples, other_samples) = all_samples.split_at(n_samples_ones);
        let zeros_samples = &other_samples[..n_samples_zeros];
        (ones_samples, zeros_samples)
    }

    pub fn split_samples_mut<'a>(index_after_l1l2: &'a mut [u64], data: Bits<&[u8]>, count_ones: u64) -> (&'a mut [u32], &'a mut [u32]) {
        let all_samples = cast_to_u32_mut(index_after_l1l2);
        let n_samples_ones = size::samples_for_bits(count_ones);
        let n_samples_zeros = size::samples_for_bits(data.used_bits() - count_ones);
        let (ones_samples, other_samples) = all_samples.split_at_mut(n_samples_ones);
        let zeros_samples = &mut other_samples[..n_samples_zeros];
        (ones_samples, zeros_samples)
    }
}

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

    let (l0_index, index_after_l0) = unpack::split_l0_mut(into, bits);
    let (l1l2_index, index_after_l1l2) = unpack::split_l1l2_mut(index_after_l0, bits);

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

    // TODO: Build the select index
    let (samples_ones, samples_zeros) =
        unpack::split_samples_mut(index_after_l1l2, bits, total_count_ones);

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

fn build_samples<W: OnesOrZeros>(l0_index: &[u64], l1l2_index: &[L1L2Entry], bits: Bits<&[u8]>, samples: &mut [u32]) {
    unimplemented!();
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


pub fn rank_ones(index: &[u64], bits: Bits<&[u8]>, idx: u64) -> Option<u64> {
    if idx >= bits.used_bits() {
        return None;
    } else if idx == 0 {
        return Some(0);
    }

    let l0_size = size::l0(bits.used_bits()) as usize;
    let (l0_index, inner_indexes) = index.split_at(l0_size);

    // Disallow future use of index by shadowing
    #[allow(unused_variables)]
    let index = ();

    let l0_idx = idx / size::BITS_PER_L0_BLOCK;
    debug_assert!(l0_idx < l0_size as u64);

    let l0_rank = if l0_idx > 0 {
        l0_index[l0_idx as usize - 1]
    } else {
        0
    };

    let inner_index_l1l2 = {
        let remaining_bits = bits.used_bits() - l0_idx * size::BITS_PER_L0_BLOCK;
        let inner_index_start = l0_idx as usize * size::INNER_INDEX_SIZE;
        let inner_l1l2_size = if remaining_bits < size::BITS_PER_L0_BLOCK {
            size::l1l2(remaining_bits)
        } else {
            size::INNER_INDEX_L1L2_SIZE
        };
        &inner_indexes[inner_index_start..inner_index_start + inner_l1l2_size]
    };
    let l0_offset = idx % size::BITS_PER_L0_BLOCK;

    // Disallow future use of inner_indexes by shadowing
    #[allow(unused_variables)]
    let inner_indexes = ();

    let block_idx = (l0_offset / size::BITS_PER_BLOCK) as usize;
    let block_offset = l0_offset % size::BITS_PER_BLOCK;

    let l1_idx = block_idx / 4;
    let l1_offset_in_blocks = block_idx % 4;
    debug_assert!(l1_idx < inner_index_l1l2.len());

    let l1l2_entry = L1L2Entry::from(inner_index_l1l2[l1_idx]);
    let l1_rank = l1l2_entry.base_rank();

    let l2_index = l1_offset_in_blocks;
    let l2_rank = if l2_index > 0 {
        l1l2_entry.sub_rank(l2_index - 1)
    } else {
        0
    };

    let block_start_bytes = l0_idx as usize * size::BYTES_PER_L0_BLOCK +
        block_idx * size::BYTES_PER_BLOCK;
    let from_block = bits.skip_bytes(block_start_bytes);
    from_block.rank::<OneBits>(block_offset).map(
        |in_block_rank| {
            l0_rank + l1_rank + l2_rank + in_block_rank
        },
    )
}

pub fn rank<W: OnesOrZeros>(index: &[u64], bits: Bits<&[u8]>, idx: u64) -> Option<u64> {
    rank_ones(index, bits, idx).map(|res_ones| W::convert_count(res_ones, idx))
}

fn index_within<T: Sized>(slice: &[T], item: &T) -> Option<usize> {
    use std::mem::size_of;

    if size_of::<T>() == 0 {
        return None;
    };

    let slice_start = (slice.as_ptr() as *const T) as usize;
    let item_pos = (item as *const T) as usize;
    if item_pos < slice_start {
        return None;
    };

    let idx = (item_pos - slice_start) / size_of::<T>();
    if idx >= slice.len() { None } else { Some(idx) }
}

pub fn select<W: OnesOrZeros>(index: &[u64], bits: Bits<&[u8]>, target_rank: u64) -> Option<u64> {
    if target_rank >= count::<W>(index, bits) {
        return None;
    }

    let l0_size = size::l0(bits.used_bits()) as usize;
    let (l0_index, inner_indexes) = index.split_at(l0_size);


    // slice library does not specify which this returns if
    // there are multiple equal counts, so it's just a hint
    let l0_idx_hint = {
        if l0_index.len() > 16 {
            let search_for = target_rank + 1;
            let res = l0_index.binary_search_by_key(&search_for, |count_ref| {
                let idx = index_within(l0_index, count_ref).unwrap();
                W::convert_count(*count_ref, (idx as u64 + 1) * size::BITS_PER_L0_BLOCK)
            });
            match res {
                Ok(i) => i,
                Err(i) => i,
            }
        } else {
            l0_index.len() - 1
        }
    };

    let mut l0_idx = l0_idx_hint;
    debug_assert!(l0_idx < l0_index.len());
    debug_assert!(
        W::convert_count(
            l0_index[l0_idx],
            (l0_idx as u64 + 1) * size::BITS_PER_L0_BLOCK,
        ) > target_rank
    );
    while l0_idx > 0 &&
        W::convert_count(
            l0_index[l0_idx - 1],
            l0_idx as u64 * size::BITS_PER_L0_BLOCK,
        ) > target_rank
    {
        l0_idx -= 1;
    }
    let l0_idx = l0_idx;
    let l0_block_rank_ones = if l0_idx > 0 { l0_index[l0_idx - 1] } else { 0 };
    let l0_block_count_ones = l0_index[l0_idx] - l0_block_rank_ones;
    let bits_before_l0_block = l0_idx as u64 * size::BITS_PER_L0_BLOCK;
    let l0_block_bits = min(
        bits.used_bits() - bits_before_l0_block,
        size::BITS_PER_L0_BLOCK,
    );
    let l0_block_rank = W::convert_count(l0_block_rank_ones, bits_before_l0_block);
    debug_assert!(l0_block_rank <= target_rank);
    let l0_block_target_rank = target_rank - l0_block_rank;

    let l1l2_size = size::l1l2(l0_block_bits);
    let inner_index_start = l0_idx as usize * size::INNER_INDEX_SIZE;
    let inner_index_l1l2 = &inner_indexes[inner_index_start..inner_index_start + l1l2_size];
    let inner_index_samples = {
        let ones_samples_size = size::sample_words_for_matching_bits(l0_block_count_ones);
        let ones_start = inner_index_start + l1l2_size;
        let zeros_start = ones_start + ones_samples_size;
        if W::is_ones() {
            &inner_indexes[ones_start..zeros_start]
        } else {
            let zeros_samples_size =
                size::sample_words_for_matching_bits(l0_block_bits - l0_block_count_ones);
            &inner_indexes[zeros_start..zeros_start + zeros_samples_size]
        }
    };
    let inner_index_samples = cast_to_u32(inner_index_samples);

    let sample_idx = (l0_block_target_rank / size::SAMPLE_LENGTH) as usize;
    let superblock_idx = inner_index_samples[sample_idx] as usize;
    let l1l2_entry = L1L2Entry::from(inner_index_l1l2[superblock_idx]);

    let bits_before_superblock = superblock_idx as u64 * size::BITS_PER_SUPERBLOCK;
    let superblock_rank = W::convert_count(l1l2_entry.base_rank(), bits_before_superblock);
    let superblock_target_rank = l0_block_target_rank - superblock_rank;

    let mut bits_before_lower_block = 0;
    let mut lower_block_rank = 0;

    for i in (0..3).rev() {
        let bits_before = (i + 1) as u64 * size::BITS_PER_BLOCK;
        let sub_rank = W::convert_count(l1l2_entry.sub_rank(i), bits_before);
        if sub_rank <= superblock_target_rank {
            bits_before_lower_block = bits_before;
            lower_block_rank = sub_rank;
            break;
        }
    }

    let final_search_target_rank = superblock_target_rank - lower_block_rank;
    let bits_before_final_search = bits_before_l0_block + bits_before_superblock +
        bits_before_lower_block;
    debug_assert_eq!(bits_before_l0_block % 8, 0);
    let bytes_before_final_search = (bits_before_final_search / 8) as usize;
    let bits_for_final_search = bits.skip_bytes(bytes_before_final_search);

    bits_for_final_search
        .select::<W>(final_search_target_rank)
        .map(|res| res + bits_before_final_search)
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
