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

use crate::bits::BitsRef;
use crate::ceil_div_u64;
use crate::ones_or_zeros::{OneBits, OnesOrZeros, ZeroBits};
use core::ops::Range;

mod size {
    use super::*;

    pub const BITS_PER_L0_BLOCK: u64 = 1 << 32;
    pub const BITS_PER_L1_BLOCK: u64 = BITS_PER_L2_BLOCK * 4;
    pub const BITS_PER_L2_BLOCK: u64 = 512;

    pub fn l0(total_bits: u64) -> usize {
        ceil_div_u64(total_bits, BITS_PER_L0_BLOCK) as usize
    }

    pub fn l1l2(total_bits: u64) -> usize {
        ceil_div_u64(total_bits, BITS_PER_L1_BLOCK) as usize
    }

    pub const L1_BLOCKS_PER_L0_BLOCK: usize = (BITS_PER_L0_BLOCK / BITS_PER_L1_BLOCK) as usize;
    pub const L2_BLOCKS_PER_L1_BLOCK: u64 = BITS_PER_L1_BLOCK / BITS_PER_L2_BLOCK;

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
        fn block_sizes_evenly_divide() {
            assert_eq!(BITS_PER_L0_BLOCK % BITS_PER_L1_BLOCK, 0);
            assert_eq!(BITS_PER_L0_BLOCK % BITS_PER_L2_BLOCK, 0);
            assert_eq!(BITS_PER_L1_BLOCK % BITS_PER_L2_BLOCK, 0);
        }
    }
}

pub struct IndexSize {
    l0_size: usize,
    l1l2_size: usize,
}

impl IndexSize {
    pub fn for_n_bits(n: u64) -> Self {
        Self {
            l0_size: size::l0(n),
            l1l2_size: size::l1l2(n),
        }
    }

    pub fn for_bits(bits: BitsRef) -> Self {
        Self::for_n_bits(bits.len())
    }

    pub fn l0_entries(&self) -> usize {
        self.l0_size
    }

    pub fn l1l2_entries(&self) -> usize {
        self.l1l2_size
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct L0Entry(u64);

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct L1L2Entry(u64);

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
struct L2Ranks([u32; 4]);

impl L0Entry {
    const ZERO: Self = Self(0);
}

impl L1L2Entry {
    const ZERO: Self = Self(0);
}

impl Default for L0Entry {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Default for L1L2Entry {
    fn default() -> Self {
        Self::ZERO
    }
}

impl L1L2Entry {
    fn pack_raw(items: [u32; 4]) -> Option<Self> {
        if items[1..].iter().any(|&x| x >= 0x0400) {
            None
        } else {
            Some(Self(
                0u64 | ((items[0] as u64) << 32)
                    | ((items[1] as u64) << 22)
                    | ((items[2] as u64) << 12)
                    | ((items[3] as u64) << 2),
            ))
        }
    }

    fn unpack_raw(self) -> [u32; 4] {
        [
            ((self.0 >> 32) & 0xffffffff) as u32,
            ((self.0 >> 22) & 0x3ff) as u32,
            ((self.0 >> 12) & 0x3ff) as u32,
            ((self.0 >> 2) & 0x3ff) as u32,
        ]
    }

    fn pack(ranks: L2Ranks) -> Option<Self> {
        let mut parts = ranks.0;
        parts[3] -= parts[2];
        parts[2] -= parts[1];
        parts[1] -= parts[0];
        Self::pack_raw(parts)
    }

    fn unpack(self) -> L2Ranks {
        let mut res = self.unpack_raw();
        res[1] += res[0];
        res[2] += res[1];
        res[3] += res[2];
        L2Ranks(res)
    }
}

pub struct IndexedBits<'a> {
    l0_index: &'a [L0Entry],
    l1l2_index: &'a [L1L2Entry],
    data: BitsRef<'a>,
}

fn zip_eq<L, R>(left: L, right: R) -> core::iter::Zip<L, R>
where L: core::iter::ExactSizeIterator,
R: core::iter::ExactSizeIterator,
{
    debug_assert_eq!(left.len(), right.len(), "Iterators are expected to have the same length");
    left.zip(right)
}

#[derive(Debug, Copy, Clone)]
pub struct IndexSizeError;

fn build_index<'a, 'b>(
    l0_index: &'a mut [L0Entry],
    l1l2_index: &'b mut [L1L2Entry],
    data: BitsRef,
) -> Result<(&'a [L0Entry], &'b [L1L2Entry]), IndexSizeError> {
    fn build_l1(l1l2_index_part: &mut [L1L2Entry], data_part: BitsRef) -> u64 {
        let l1_chunks = data_part
            .chunks(size::BITS_PER_L1_BLOCK)
            .expect("The chunk size should not be zero");
        debug_assert_eq!(l1l2_index_part.len(), l1_chunks.len());
        zip_eq(l1l2_index_part.iter_mut(), l1_chunks)
            .for_each(|(entry, data_part)| {
                let mut parts = [0; 4];
                let l2_chunks = data_part
                    .chunks(size::BITS_PER_L2_BLOCK)
                    .expect("The chunk size should not be zero");
                debug_assert!(parts.len() >= l2_chunks.len());
                parts
                    .iter_mut()
                    .zip(l2_chunks)
                    .for_each(|(write_count, data_part)| {
                        let count = data_part.count_ones();
                        debug_assert!(count < 0x400 as u64);
                        *write_count = count as u32;
                    });
                *entry = L1L2Entry::pack_raw(parts).expect("There aren't enough ");
            });
        let mut running_total = 0u64;
        l1l2_index_part.iter_mut().for_each(|entry| {
            let part_counts = entry.unpack_raw();
            let mut ranks = L2Ranks([0; 4]);
            debug_assert!(running_total < u32::max_value() as u64);
            ranks.0[0] = running_total as u32;
            ranks.0[1] = ranks.0[0] + part_counts[0];
            ranks.0[2] = ranks.0[1] + part_counts[1];
            ranks.0[3] = ranks.0[2] + part_counts[2];
            *entry = L1L2Entry::pack(ranks).expect("All of the counts should be small enough");
            running_total = ranks.0[3] as u64 + part_counts[3] as u64;
        });
        running_total as u64
    }

    let total_bits = data.len();
    let l0_index = match l0_index.get_mut(..size::l0(total_bits)) {
        Some(x) => x,
        None => return Err(IndexSizeError),
    };
    let l1l2_index = match l1l2_index.get_mut(..size::l1l2(total_bits)) {
        Some(x) => x,
        None => return Err(IndexSizeError),
    };

    let l0_l1l2_chunks = l1l2_index.chunks_mut(size::L1_BLOCKS_PER_L0_BLOCK);
    let l0_data_chunks = data
        .chunks(size::BITS_PER_L0_BLOCK)
        .expect("The chunk size should not be zero");
    zip_eq(l0_index.iter_mut(), zip_eq(l0_l1l2_chunks, l0_data_chunks))
        .for_each(|(write_count, (l1l2_part, data_part))| {
            *write_count = L0Entry(build_l1(l1l2_part, data_part))
        });

    let mut running_total = 0;
    l0_index.iter_mut().for_each(|entry| {
        running_total += entry.0;
        entry.0 = running_total;
    });
    Ok((l0_index, l1l2_index))
}

fn midpoint(a: u64, b: u64) -> u64 {
    #[cfg(debug_assertions)]
    fn check_midpoint(a: u64, b: u64, mid: u64) -> bool {
        let min = core::cmp::min(a, b);
        let max = core::cmp::max(a, b);
        mid >= min && mid <= max
        && max - (mid - min + mid) <= 1
    }

    let a_hi = a >> 1;
    let b_hi = b >> 1;
    let a_lo = a & 1;
    let b_lo = b & 1;
    let lo = a_lo & b_lo;
    debug_assert!(a_hi.checked_add(b_hi).is_some());
    let hi = a_hi.wrapping_add(b_hi);
    debug_assert!(hi.checked_add(lo).is_some());
    let mid = hi.wrapping_add(lo);
    debug_assert!(check_midpoint(a, b, mid));
    mid
}

fn binary_search(
    mut in_range: Range<u64>,
    is_greater_or_equal_tgt: impl Fn(u64) -> bool,
) -> u64 {
    assert!(in_range.start <= in_range.end, "Range start bound must not be after end bound");

    const MIN_SEARCH_SIZE: u64 = 16;
    while in_range.end.wrapping_sub(in_range.start) > MIN_SEARCH_SIZE {
        let mid = midpoint(in_range.start, in_range.end);
        if is_greater_or_equal_tgt(mid) {
            in_range.end = mid;
        } else {
            in_range.start = mid + 1;
        }
    }

    for idx in in_range.clone() {
        if is_greater_or_equal_tgt(idx) {
            in_range.end = idx;
            break;
        } else {
            in_range.start = idx + 1;
        }
    }

    debug_assert!(in_range.start >= in_range.end && in_range.start - in_range.end <= 1);
    in_range.start
}

impl<'a> IndexedBits<'a> {
    pub(crate) unsafe fn from_existing_index(
        l0_index: &'a [L0Entry],
        l1l2_index: &'a [L1L2Entry],
        data: BitsRef<'a>,
    ) -> Self {
        Self {
            l0_index,
            l1l2_index,
            data,
        }
    }

    pub fn from_bits(
        l0_index_space: &'a mut [L0Entry],
        l1l2_index_space: &'a mut [L1L2Entry],
        data: BitsRef<'a>,
    ) -> Result<Self, IndexSizeError> {
        let (l0_index, l1l2_index) = build_index(l0_index_space, l1l2_index_space, data)?;
        Ok(unsafe { Self::from_existing_index(l0_index, l1l2_index, data) })
    }

    pub fn len(&self) -> u64 {
        self.data.len()
    }

    pub fn count_ones(&self) -> u64 {
        self.l0_index.last().map_or(0, |entry| entry.0)
    }

    pub fn count_zeros(&self) -> u64 {
        ZeroBits::convert_count(self.count_ones(), self.len())
    }

    fn read_rank_ones_from_idx(&self, l2_blocks_from_start: u64) -> Option<u64> {
        let l1_blocks_from_start = (l2_blocks_from_start / size::L2_BLOCKS_PER_L1_BLOCK) as usize;
        let &l1l2_entry = self.l1l2_index.get(l1_blocks_from_start)?;

        let l2_idx = (l2_blocks_from_start % size::L2_BLOCKS_PER_L1_BLOCK) as usize;
        let l0_idx = l1_blocks_from_start / size::L1_BLOCKS_PER_L0_BLOCK;

        let l0_rank = if l0_idx == 0 {
            0
        } else {
            let get_idx = l0_idx.wrapping_sub(1);
            debug_assert!(self.l0_index.get(get_idx).is_some());
            unsafe { self.l0_index.get_unchecked(get_idx) }.0
        };

        let l1l2_rank = {
            let l2_index = l1l2_entry.unpack();
            debug_assert!(l2_index.0.get(l2_idx).is_some());
            *unsafe { l2_index.0.get_unchecked(l2_idx) }
        };

        Some(l0_rank + l1l2_rank as u64)
    }

    pub fn rank_ones(&self, idx: u64) -> Option<u64> {
        if idx >= self.len() {
            None
        } else {
            let l2_blocks_from_start = idx / size::BITS_PER_L2_BLOCK;

            let rank_from_data = {
                let l2_block_start_idx = l2_blocks_from_start * size::BITS_PER_L2_BLOCK;
                let from_l2_block_idx = idx.checked_sub(l2_block_start_idx)
                    .expect("It's impossible for the l2 block to start after the target position");
                let (_data_upto_index_pos, data_from_index_pos) =
                    self.data.split_at(l2_block_start_idx)
                        .expect("If the index is not in range it's a bug");
                data_from_index_pos.rank_ones(from_l2_block_idx)
                    .expect("If the index is not in range it's a bug")
            };

            let rank_from_index =
                self.read_rank_ones_from_idx(l2_blocks_from_start)
                    .expect("If the index is not in range it's a bug");

            Some(rank_from_index + rank_from_data)
        }
    }

    pub fn rank_zeros(&self, idx: u64) -> Option<u64> {
        self.rank_ones(idx)
            .map(|rank_ones| ZeroBits::convert_count(rank_ones, idx))
    }

    fn read_select_hint_from_index<W: OnesOrZeros>(&self, target_rank: u64) -> Option<(u64, u64)> {
        if target_rank >= W::convert_count(self.count_ones(), self.len()) {
            return None;
        }

        fn read_rank_from_idx_or_bug<W: OnesOrZeros>(indexed: &IndexedBits, l2_blocks_from_start: u64) -> u64 {
            let rank_ones = indexed.read_rank_ones_from_idx(l2_blocks_from_start)
                .expect("If the index is not in range it's a bug");
            W::convert_count(rank_ones, l2_blocks_from_start * size::BITS_PER_L2_BLOCK)
        }

        let l2_blocks_range = 0..(self.data.len() / size::BITS_PER_L2_BLOCK);
        let l2_block_with_higher_rank =
            binary_search(l2_blocks_range, |l2_blocks_from_start| {
                read_rank_from_idx_or_bug::<W>(self, l2_blocks_from_start) > target_rank
            });
        let l2_block_to_search_from =
            l2_block_with_higher_rank.checked_sub(1)
            .expect("The first block must have rank 0, so it can't have a higher rank");
        let l2_block_rank = read_rank_from_idx_or_bug::<W>(self, l2_block_to_search_from);
        let idx_to_search_from = l2_block_to_search_from * size::BITS_PER_L2_BLOCK;
        let search_rank = target_rank.checked_sub(l2_block_rank).expect("If rank of hint is too high that's a bug");
        Some((idx_to_search_from, search_rank))
    }

    pub fn select_ones(&self, rank: u64) -> Option<u64> {
        let (idx_to_search_from, further_search_rank) =
            self.read_select_hint_from_index::<OneBits>(rank)?;
        let (_data_to_ignore, data_to_search) = self.data.split_at(idx_to_search_from)
            .expect("If the index of hint is out of range that's a bug");
        let search_result =
            data_to_search.select_ones(further_search_rank).expect("If the search rank is out of range that's a bug");
        Some(idx_to_search_from + search_result)
    }

    pub fn select_zeros(&self, rank: u64) -> Option<u64> {
        let (idx_to_search_from, further_search_rank) =
            self.read_select_hint_from_index::<ZeroBits>(rank)?;
        let (_data_to_ignore, data_to_search) = self.data.split_at(idx_to_search_from)
            .expect("If the index of hint is out of range that's a bug");
        let search_result =
            data_to_search.select_zeros(further_search_rank).expect("If the search rank is out of range that's a bug");
        Some(idx_to_search_from + search_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::Bits;
    use crate::bits::tests::prepared_bits;
    use std::vec::Vec;

    impl IndexSize {
        fn l0_vec(&self) -> Vec<L0Entry> {
            vec![L0Entry::default(); self.l0_entries()]
        }

        fn l1l2_vec(&self) -> Vec<L1L2Entry> {
            vec![L1L2Entry::default(); self.l1l2_entries()]
        }
    }

    #[test]
    fn select_bug_issue_15() {
        // When the bit we are selecting is in the same block as the next index sample
        let mut data = vec![0xffu8; 8192 / 8 * 2];
        data[8192 / 8 - 1] = 0;
        let data = Bits::from(&data[..], 0, 8192 * 2).unwrap();
        let size = IndexSize::for_bits(data);
        let mut l0 = size.l0_vec();
        let mut l1l2 = size.l1l2_vec();
        let index =
            IndexedBits::from_bits(l0.as_mut_slice(), l1l2.as_mut_slice(), data)
                .unwrap();
        assert_eq!(index.select_ones(8191), Some(8199));
    }

    #[test]
    fn small_indexed_tests() {
        use rand::{Rng, RngCore, SeedableRng};
        use rand_xorshift::XorShiftRng;
        let n_bits: u64 = (1 << 19) - 1;
        let n_bytes: usize = ceil_div_u64(n_bits, 8) as usize;
        let seed = [
            42, 73, 197, 231, 255, 43, 87, 05, 50, 13, 74, 107, 195, 231, 5, 1,
        ];
        let mut rng = XorShiftRng::from_seed(seed);
        let data = {
            let mut data = vec![0u8; n_bytes];
            rng.fill_bytes(&mut data);
            data
        };
        let data = Bits::from(&data[..], 0, n_bits).expect("Should have enough bytes");
        let size = IndexSize::for_bits(data);
        let mut l0 = size.l0_vec();
        let mut l1l2 = size.l1l2_vec();
        let index = IndexedBits::from_bits(l0.as_mut_slice(), l1l2.as_mut_slice(), data).unwrap();

        let expected_count_ones = data.count_ones();
        let expected_count_zeros = n_bits - expected_count_ones;
        assert_eq!(expected_count_ones, index.count_ones());
        assert_eq!(expected_count_zeros, index.count_zeros());

        assert_eq!(None, index.rank_ones(n_bits));
        assert_eq!(None, index.rank_zeros(n_bits));

        let mut gen_sorted_in = |range: core::ops::Range<u64>| {
            let mut r: Vec<u64> = (0..1000)
                .map(|_| rng.gen_range(range.start, range.end))
                .collect();
            r.sort_unstable();
            r
        };

        let rank_idxs = gen_sorted_in(0..n_bits);
        for idx in rank_idxs {
            assert_eq!(data.rank_ones(idx), index.rank_ones(idx));
            assert_eq!(data.rank_zeros(idx), index.rank_zeros(idx));
        }

        assert_eq!(None, index.select_ones(expected_count_ones));
        let one_ranks = gen_sorted_in(0..expected_count_ones);
        for rank in one_ranks {
            assert_eq!(data.select_ones(rank), index.select_ones(rank));
        }

        assert_eq!(None, index.select_zeros(expected_count_zeros));
        let zero_ranks = gen_sorted_in(0..expected_count_zeros);
        for rank in zero_ranks {
            assert_eq!(data.select_zeros(rank), index.select_zeros(rank));
        }
    }
}
