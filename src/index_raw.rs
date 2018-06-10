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

use super::ceil_div_u64;
use bits_type::Bits;
use ones_or_zeros::{OneBits, OnesOrZeros};

mod size {
    use super::*;

    pub const BITS_PER_L0_BLOCK: u64 = 1 << 32;
    pub const BITS_PER_BLOCK: u64 = 512;

    pub const BITS_PER_SUPERBLOCK: u64 = BITS_PER_BLOCK * 4;

    pub const BYTES_PER_BLOCK: usize = (BITS_PER_BLOCK / 8) as usize;
    pub const BYTES_PER_L0_BLOCK: usize = (BITS_PER_L0_BLOCK / 8) as usize;

    pub const BYTES_PER_SUPERBLOCK: usize = BYTES_PER_BLOCK * 4;

    pub fn l0(total_bits: u64) -> usize {
        ceil_div_u64(total_bits, BITS_PER_L0_BLOCK) as usize
    }

    pub fn l1l2(total_bits: u64) -> usize {
        ceil_div_u64(total_bits, BITS_PER_SUPERBLOCK) as usize
    }

    fn rank_index(total_bits: u64) -> usize {
        l0(total_bits) + l1l2(total_bits)
    }

    pub const SAMPLE_LENGTH: u64 = 8192;

    pub fn sample_words(total_bits: u64) -> usize {
        ceil_div_u64(total_bits, SAMPLE_LENGTH * 2) as usize + l0(total_bits)
    }

    pub fn total_index_words(total_bits: u64) -> usize {
        rank_index(total_bits) + sample_words(total_bits)
    }

    pub const INNER_INDEX_L1L2_SIZE: usize = (BITS_PER_L0_BLOCK / BITS_PER_SUPERBLOCK) as usize;
    pub const INNER_INDEX_SIZE: usize = INNER_INDEX_L1L2_SIZE +
        (BITS_PER_L0_BLOCK / (SAMPLE_LENGTH * 2) + 1) as usize;

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn bytes_evenly_divide_block_sizes() {
            assert_eq!(BYTES_PER_BLOCK, ceil_div_u64(BITS_PER_BLOCK, 8) as usize);
            assert_eq!(BYTES_PER_BLOCK, (BITS_PER_BLOCK / 8) as usize);
            assert_eq!(
                BYTES_PER_L0_BLOCK,
                ceil_div_u64(BITS_PER_L0_BLOCK, 8) as usize
            );
            assert_eq!(BYTES_PER_L0_BLOCK, (BITS_PER_L0_BLOCK / 8) as usize);
        }

        #[test]
        fn l1l2_evenly_divide_l0() {
            // This property is needed so that the size of the l1l2
            // index works out correctly if calculated across separate
            // l0 blocks.
            assert_eq!(0, BITS_PER_L0_BLOCK % BITS_PER_BLOCK);
            assert_eq!(0, (BITS_PER_L0_BLOCK / BITS_PER_BLOCK) % 4);
        }

        #[test]
        fn samples_evenly_divide_l0() {
            // This property is needed so that the size of the sampling
            // index works out correctly if calculated across separate
            // l0 blocks.
            assert_eq!(0, BITS_PER_L0_BLOCK % SAMPLE_LENGTH);
            assert_eq!(0, (BITS_PER_L0_BLOCK / SAMPLE_LENGTH) % 2);
            assert_eq!(1, l0(BITS_PER_L0_BLOCK));
        }

        #[test]
        fn inner_index_size() {
            assert_eq!(
                l1l2(BITS_PER_L0_BLOCK) + sample_words(BITS_PER_L0_BLOCK),
                INNER_INDEX_SIZE
            );
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct L1L2Entry(u64);

fn cast_to_l1l2<'a>(data: &'a [u64]) -> &'a [L1L2Entry] {
    use std::mem::{size_of, align_of};
    debug_assert_eq!(size_of::<u64>(), size_of::<L1L2Entry>());
    debug_assert_eq!(align_of::<u64>(), align_of::<L1L2Entry>());

    unsafe {
        use std::slice::from_raw_parts;
        let n = data.len();
        let ptr = data.as_ptr() as *const L1L2Entry;
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

// First pass of building the index, where each entry holds its total size,
// but the sub-counts are fully built in this pass.
fn build_inner_proto_index(l1l2_index: &mut [L1L2Entry], data: Bits<&[u8]>) {
    debug_assert!(data.used_bits() > 0);
    debug_assert!(data.used_bits() <= size::BITS_PER_L0_BLOCK);
    debug_assert!(l1l2_index.len() == size::l1l2(data.used_bits()));

    data.chunks_bytes(size::BYTES_PER_SUPERBLOCK)
        .zip(l1l2_index.iter_mut())
    // This loop could be parallelised
        .for_each(|(quad_chunk, write_to)| {
            let mut counts = [0u16; 4];
            quad_chunk
                .chunks_bytes(size::BYTES_PER_BLOCK)
                .zip(counts.iter_mut())
                .for_each(|(chunk, write_to)| {
                    *write_to = chunk.count::<OneBits>() as u16
                });

            counts[1] += counts[0];
            counts[2] += counts[1];
            counts[3] += counts[2];

            *write_to = L1L2Entry::pack(counts[3] as u32, [counts[0], counts[1], counts[2]]);
        });
}

fn build_inner_select_index_part<I>(select_index: &mut [u32], superblock_rank_and_count: I)
where
    I: Iterator<Item = (u32, u32)> + Clone,
{
    debug_assert!(size::SAMPLE_LENGTH >= size::BITS_PER_SUPERBLOCK);

    let chosen_superblock_idxs = superblock_rank_and_count.enumerate().filter_map(
        |(superblock_idx,
          (base_rank,
           count))| {
            let target_rank = {
                let remainder = base_rank % (size::SAMPLE_LENGTH as u32);
                if remainder == 0 {
                    0
                } else {
                    (size::SAMPLE_LENGTH as u32) - remainder
                }
            };
            if target_rank < count {
                Some(superblock_idx as u32)
            } else {
                None
            }
        },
    );

    debug_assert!(chosen_superblock_idxs.clone().count() <= select_index.len());

    for (superblock_idx, write_to) in chosen_superblock_idxs.zip(select_index.iter_mut()) {
        *write_to = superblock_idx;
    }
}

/// Will ONLY work if the L1L2 is in the proto-index state
/// (where it has count within superblock in place of the base_rank)
fn build_inner_select_index(select_index: &mut [u64], proto_l1l2_index: &[L1L2Entry]) {
    let total_count: u64 = proto_l1l2_index
        .iter()
        .map(|entry| entry.base_rank() as u64)
        .sum();
    let (select_ones_index, select_zeros_index) =
        select_index.split_at_mut(ceil_div_u64(total_count, size::SAMPLE_LENGTH * 2) as usize);
    let select_ones_index = cast_to_u32_mut(select_ones_index);
    let select_zeros_index = cast_to_u32_mut(select_zeros_index);

    // These two operations could be done in parallel
    build_inner_select_index_part(
        select_ones_index,
        proto_l1l2_index.iter().scan(
            0u64,
            |running_base_rank, entry| {
                let count = entry.base_rank();
                let base_rank = running_base_rank.clone();
                *running_base_rank = base_rank + count;
                Some((base_rank as u32, count as u32))
            },
        ),
    );
    build_inner_select_index_part(
        select_zeros_index,
        proto_l1l2_index.iter().scan(
            0u64,
            |running_base_rank, entry| {
                let count = size::BITS_PER_SUPERBLOCK - entry.base_rank();
                let base_rank = running_base_rank.clone();
                *running_base_rank = base_rank + count;
                Some((base_rank as u32, count as u32))
            },
        ),
    );
}

/// Returns the total set bit count
fn build_inner_index(index: &mut [u64], data: Bits<&[u8]>) -> u64 {
    debug_assert!(data.used_bits() > 0);
    debug_assert!(data.used_bits() <= size::BITS_PER_L0_BLOCK);
    debug_assert!(
        index.len() == size::l1l2(data.used_bits()) + size::sample_words(data.used_bits())
    );

    let (proto_l1l2_index, select_index) = index.split_at_mut(size::l1l2(data.used_bits()));
    let proto_l1l2_index = cast_to_l1l2_mut(proto_l1l2_index);

    build_inner_proto_index(proto_l1l2_index, data);
    build_inner_select_index(select_index, proto_l1l2_index);

    // Pass through reassigning each entry to hold its rank to finish.
    let mut running_rank = 0u64;
    for entry in proto_l1l2_index.iter_mut() {
        let base_rank = running_rank.clone();
        running_rank += entry.base_rank();
        entry.set_base_rank(base_rank as u32);
    }

    running_rank
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

    let l0_size = size::l0(bits.used_bits());
    let (l0_index, index) = into.split_at_mut(l0_size);

    // This loop could be parallelised
    bits.chunks_bytes(size::BYTES_PER_L0_BLOCK)
        .zip(index.chunks_mut(size::INNER_INDEX_SIZE))
        .zip(l0_index.iter_mut())
        .for_each(|((l0_chunk, inner_index), write_to)| {
            *write_to = build_inner_index(inner_index, l0_chunk)
        });

    let mut total_count = 0u64;
    for l0_index_cell in l0_index.iter_mut() {
        total_count += *l0_index_cell;
        *l0_index_cell = total_count;
    }

    Ok(())
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

/*

fn index_within<T: Sized>(slice: &[T], item: &T) -> Option<usize> {
    use std::mem::size_of;

    if size_of::<T>() == 0 {
        return None;
    };

    let slice_start = (slice as *const T) as usize;
    let item_pos = (item as *const T) as usize;
    if item_pos < slice_start {
        return None;
    };

    let idx = (item_pos - slice_start) / size_of::<T>();
    if idx >= slice.len() { None } else { Some(idx) }
}

pub fn select_unchecked<W: OnesOrZeros>(
    index: &[u64],
    bits: Bits<&[u8]>,
    target_rank: u64,
) -> Option<u64> {
    let l0_size = size::l0(bits.used_bits());
    let (l0_index, rest) = index.split_at(l0_size);

    // TODO: This can be made into a binary search
    let l0_block_index = l0_index.iter().enumerate().position(|(i, &c)| {
        let total = (i as u64 + 1) * size::BITS_PER_L0_BLOCK;
        let total = min(total, bits.used_bits());
        W::convert_count(c, total) > target_rank
    });
    let l0_block_index = match l0_block_index {
        None => return None,
        Some(i) => i,
    };

    let inner_index = &index[l0_size + l0_block_index * size::INNER_INDEX_SIZE..];
    let inner_index = &inner_index[..min(inner_index.len(), size::INNER_INDEX_SIZE)];


    panic!("Not implemented")
}

*/
