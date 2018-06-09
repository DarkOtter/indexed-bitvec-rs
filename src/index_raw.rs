//! This module contains the raw functions for the index rank
//! and select operations, as well as building indexes.
//!
//! The functions here do minimal if any checking on the size
//! or validity of indexes vs. the bitvectors they are used with,
//! so you may run into panics from e.g. out of bounds accesses
//! to slices. They should all be safe though.

use super::{ceil_div, ceil_div_u64};
use std::cmp::min;
use bits_type::Bits;
use ones_or_zeros::{OneBits, ZeroBits, OnesOrZeros};

mod size {
    use super::ceil_div_u64;

    pub const BITS_PER_L0_BLOCK: u64 = 1 << 32;
    pub const BITS_PER_BLOCK: u64 = 512;

    pub const BYTES_PER_BLOCK: usize = (BITS_PER_BLOCK / 8) as usize;
    pub const BYTES_PER_L0_BLOCK: usize = (BITS_PER_L0_BLOCK / 8) as usize;

    pub fn l0(total_bits: u64) -> usize {
        ceil_div_u64(total_bits, BITS_PER_L0_BLOCK) as usize
    }

    pub fn l1l2(total_bits: u64) -> usize {
        ceil_div_u64(total_bits, BITS_PER_BLOCK * 4) as usize
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

    pub const INNER_INDEX_SIZE: usize = (BITS_PER_L0_BLOCK / (BITS_PER_BLOCK * 4)) as usize +
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

mod index_types {
    use super::size;

    #[derive(Copy, Clone, Debug)]
    pub struct L1L2Entry(u64);
    #[derive(Copy, Clone, Debug)]
    pub struct InnerIndexRef<'a>(&'a [u64]);
    #[derive(Copy, Clone, Debug)]
    pub struct IndexRef<'a>(&'a [u64]);

    impl L1L2Entry {
        pub fn pack(base_rank: u32, sub_ranks: [u16; 3]) -> Self {
            L1L2Entry(
                ((base_rank as u64) << 32) | ((sub_ranks[0] as u64) << 22) |
                    ((sub_ranks[1] as u64) << 12) | ((sub_ranks[2] as u64) << 2),
            )
        }

        pub fn base_rank(self) -> u64 {
            self.0 >> 32
        }

        pub fn fset_base_rank(self, base_rank: u32) -> Self {
            L1L2Entry(((base_rank as u64) << 32) | self.0 & 0xffffffff)
        }

        pub fn sub_rank(self, i: usize) -> u64 {
            let shift = 22 - i * 10;
            (self.0 >> shift) & 0x3ff
        }
    }

    impl From<L1L2Entry> for u64 {
        fn from(entry: L1L2Entry) -> Self {
            entry.0
        }
    }

    fn inner_split_idx(total_bits: u64) -> usize {
        size::l1l2(total_bits)
    }

    fn cast_to_l1l2(data: &[u64]) -> &[L1L2Entry] {
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

    impl<'a> InnerIndexRef<'a> {
        pub fn rank_part(self, total_bits: u64) -> &'a [L1L2Entry] {
            cast_to_l1l2(&self.0[..inner_split_idx(total_bits)])
        }
    }
}

/*

/// Returns the total set bit count
fn build_inner_rank_index(index: &mut [u64], data: Bits<&[u8]>) -> u32 {
    debug_assert!(data.used_bits() > 0);
    debug_assert!(data.used_bits() <= size::BITS_PER_L0_BLOCK);
    debug_assert!(index.len() == size::l1l2(data.used_bits()));

    // This outer loop could be parallelised
    data.chunks_bytes(size::BYTES_PER_BLOCK * 4)
        .zip(index.iter_mut())
        .for_each(|(quad_chunk, write_to)| {
            let mut counts = [0u16; 4];
            quad_chunk
                .chunks_bytes(size::BYTES_PER_BLOCK)
                .zip(counts.iter_mut())
                .for_each(|(chunk, write_to)| {
                    *write_to = chunk.count::<OneBits>() as u16
                });

            for i in 1..counts.len() {
                counts[i] += counts[i - 1];
            }

            *write_to = L1L2Entry::pack(counts[3] as u32, [counts[0], counts[1], counts[2]]).into();
        });

    let mut running_total = 0u32;
    for entry in index.iter_mut() {
        let original_entry = L1L2Entry::from(entry.clone());
        *entry = original_entry.set_base_rank(running_total).into();
        running_total += original_entry.base_rank() as u32;
    }

    running_total
}

/// Used for building the select index
fn select_with_inner_rank_index<W: OnesOrZeros>(
    index: &[u64],
    data: Bits<&[u8]>,
    target_rank: u32,
) -> Option<u32> {
    debug_assert!(data.used_bits() > 0);
    debug_assert!(data.used_bits() <= size::BITS_PER_L0_BLOCK);
    debug_assert!(index.len() == size::l1l2(data.used_bits()));

    // Try to find a place we can search from,
    // it doesn't have to be exact as long as we don't skip too much.
    let mut l1_index: usize = 0;
    let mut index_entry = L1L2Entry::from(0);
    let mut index_entry_base_rank: u32 = 0;
    loop {
        if l1_index >= index.len() {
            if l1_index > 0 {
                l1_index -= 1;
                break;
            } else {
                return None;
            }
        };
        index_entry = L1L2Entry::from(index[l1_index]);
        index_entry_base_rank = W::convert_count(
            index_entry.base_rank(),
            l1_index as u64 * (size::BITS_PER_BLOCK * 4),
        ) as u32;
        if index_entry_base_rank > target_rank {
            debug_assert!(l1_index > 0);
            l1_index -= 1;
            break;
        } else if index_entry_base_rank == target_rank {
            break;
        } else {
            let diff = target_rank - index_entry_base_rank;
            let skip = (((diff - 1) as u64) / (size::BITS_PER_BLOCK * 4)) + 1;
            l1_index += skip as usize;
        }
    }

    let l1_index = l1_index;
    let index_entry = index_entry;
    let index_entry_base_rank = index_entry_base_rank;

    debug_assert!(
        (W::convert_count(
            L1L2Entry::from(index[l1_index]).base_rank(),
            l1_index as u64 * (size::BITS_PER_BLOCK * 4),
        ) as u32) <= target_rank,
        "Skipped too far for select!"
    );

    let mut start_pos = l1_index * (size::BYTES_PER_BLOCK * 4);
    let mut offset_target = target_rank - index_entry_base_rank;

    // TODO: Unroll this loop?
    let mut sub_index = 3;
    for i in 0..3 {
        let sub_rank = W::convert_count(index_entry.sub_rank(i), i as u64 * size::BITS_PER_BLOCK);
        if offset_target < sub_rank {
            sub_index = i;
            break;
        }
    }

    start_pos += sub_index as u64 * size::BYTES_PER_BLOCK;
    if sub_index > 0 {
        offset_target -= index_entry.sub_rank(sub_index - 1);
    }

    data.skip_bytes(start_pos)
        .select::<W>(offset_target as u64)
        .map(|x| x as u32)
}

fn pack_samples_index<F: Fn(u32) -> Option<u32>>(
    samples_index: &mut [u64],
    n_samples: usize,
    select: F,
) {
    debug_assert!(samples_index.len() * 2 >= n_samples);
    let n_full_packs = n_samples / 2;
    let (full_packs, part_packs) = samples_index.split_at_mut(n_full_packs);

    full_packs.iter_mut().enumerate().for_each(
        |(full_idx, write_to)| {
            let samples_idx = full_idx * 2;
            let first_sample_offset = (samples_idx as u64 * size::SAMPLE_LENGTH) as u32;
            let high_part = select(first_sample_offset).expect("Should not overflow") as u64;
            let low_part = select(first_sample_offset + size::SAMPLE_LENGTH as u32)
                .expect("Should not overflow") as u64;

            *write_to = (high_part << 32) | low_part
        },
    );

    if (n_samples % 2) != 0 {
        let part_pack = &mut part_packs[0];
        let samples_idx = n_samples - 1;
        let first_sample_offset = (samples_idx as u64 * size::SAMPLE_LENGTH) as u32;
        let high_part = select(first_sample_offset).expect("Should not overflow") as u64;
        *part_pack = high_part << 32;
    }
}

/// Returns the total set bit count
fn build_inner_index(index: &mut [u64], data: Bits<&[u8]>) -> u32 {
    debug_assert!(data.used_bits() > 0);
    debug_assert!(data.used_bits() <= size::BITS_PER_L0_BLOCK);
    let (l1l2_part, select_part) = index.split_at_mut(size::l1l2(data.used_bits()));
    debug_assert!(l1l2_part.len() == size::l1l2(data.used_bits()));
    debug_assert!(select_part.len() == size::sample_words(data.used_bits()));

    let count_ones = build_inner_rank_index(l1l2_part, data) as u64;
    let count_zeros = data.used_bits() - count_ones;

    let ones_samples = ceil_div_u64(count_ones, size::SAMPLE_LENGTH) as usize;
    let zeros_samples = ceil_div_u64(count_zeros, size::SAMPLE_LENGTH) as usize;

    let ones_sample_words = ceil_div(ones_samples, 2);
    let zeros_sample_words = ceil_div(zeros_samples, 2);

    debug_assert!(ones_sample_words + zeros_sample_words <= select_part.len());
    debug_assert!(ones_sample_words + zeros_sample_words + 1 >= select_part.len());

    let (ones_select_part, zeros_select_part) = select_part.split_at_mut(ones_sample_words);
    let zeros_select_part = &mut zeros_select_part[..zeros_sample_words];

    pack_samples_index(ones_select_part, ones_samples, |target_rank| {
        select_with_inner_rank_index::<OneBits>(l1l2_part, data, target_rank)
    });
    pack_samples_index(zeros_select_part, ones_samples, |target_rank| {
        select_with_inner_rank_index::<ZeroBits>(l1l2_part, data, target_rank)
    });

    count_ones as u32
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
            *write_to = build_inner_index(inner_index, l0_chunk) as u64
        });

    let mut total_count = 0u64;
    for l0_index_cell in l0_index.iter_mut() {
        total_count += *l0_index_cell;
        *l0_index_cell = total_count;
    }

    Ok(())
}

pub fn count_ones_unchecked(index: &[u64], bits: Bits<&[u8]>) -> u64 {
    let l0_size = size::l0(bits.used_bits());
    if bits.used_bits() == 0 {
        return 0;
    }
    debug_assert!(l0_size > 0);
    index[l0_size - 1]
}

pub fn count_unchecked<W: OnesOrZeros>(index: &[u64], bits: Bits<&[u8]>) -> u64 {
    W::convert_count(count_ones_unchecked(index, bits), bits.used_bits())
}

pub fn rank_ones_unchecked(index: &[u64], bits: Bits<&[u8]>, idx: u64) -> Option<u64> {
    if idx >= bits.used_bits() {
        return None;
    } else if idx == 0 {
        return Some(0);
    }

    let l0_size = size::l0(bits.used_bits());
    let l0_index = (idx / size::BITS_PER_L0_BLOCK) as usize;
    debug_assert!(l0_index < l0_size);

    let l0_rank = if l0_index > 0 { index[l0_index - 1] } else { 0 };
    let inner_index = &index[l0_size + l0_index * size::INNER_INDEX_SIZE..];
    let l0_offset = idx % size::BITS_PER_L0_BLOCK;

    let l1_index = (l0_offset / size::BITS_PER_BLOCK) as usize;
    debug_assert!(l1_index < inner_index.len());

    let l1l2_entry = L1L2Entry::from(inner_index[l1_index]);
    let l1_rank = l1l2_entry.base_rank();

    let l2_index = l0_offset % size::BITS_PER_BLOCK;
    let mut l2_rank = 0;
    if l2_index >= 1 {
        l2_rank += l1l2_entry.sub_count_1();
        if l2_index >= 2 {
            l2_rank += l1l2_entry.sub_count_2();
            if l2_index >= 3 {
                l2_rank += l1l2_entry.sub_count_3();
            }
        }
    }
    let l2_rank = l2_rank;

    Some(l0_rank + l1_rank + l2_rank)
}

pub fn rank_unchecked<W: OnesOrZeros>(index: &[u64], bits: Bits<&[u8]>, idx: u64) -> Option<u64> {
    rank_ones_unchecked(index, bits, idx).map(|res_ones| W::convert_count(res_ones, idx))
}

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
