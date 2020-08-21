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

use super::bits_ref::BitsRef;
use super::ceil_div_u64;
use super::ones_or_zeros::{OneBits, OnesOrZeros, ZeroBits};

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

    pub fn total_index_words(total_bits: u64) -> usize {
        l0(total_bits) + l1l2(total_bits)
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
        fn size_of_index_for_zero() {
            assert_eq!(0, total_index_words(0));
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
        if items[1..].any(|&x| x >= 0x0400) {
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

struct IndexSizeError;

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
        l1l2_index_part
            .iter_mut()
            .zip_eq(l1_chunks)
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
    debug_assert_eq!(l0_index.len(), l0_data_chunks.len());
    debug_assert_eq!(l0_index.len(), l0_l1l2_chunks.len());
    l0_index
        .iter_mut()
        .zip_eq(l0_l1l2_chunks.zip_eq(l0_data_chunks))
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

fn ternary_search_with_index<T>(
    data: &[T],
    in_which_part: impl Fn(usize, &T) -> core::cmp::Ordering,
) -> (usize, usize) {
    const LINEAR_SEARCH_SIZE: usize = 16;
    let mut low_idx = 0;
    let mut high_idx = data.len();

    loop {
        debug_assert!(high_idx.checked_sub(low_idx).is_some());
        let span = high_idx.wrapping_sub(low_idx);
        if span <= LINEAR_SEARCH_SIZE {
            break;
        }

        let offset = (span / 2).wrapping_add((span & low_idx) & 0x1);
        debug_assert!(low_idx.checked_add(offset).is_some());
        let mid = low_idx.wrapping_add(offset);

        debug_assert!(data.get(mid).is_some());
        let item = unsafe { data.get_unchecked(mid) };
        use core::cmp::Ordering;
        match in_which_part(mid, item) {
            Ordering::Less => low_idx = mid + 1,
            Ordering::Greater => high_idx = mid,
            Ordering::Equal => {
                let lower_part = &data[low_idx..mid];
                let upper_part = &data[mid + 1..high_idx];
                let (lower_idx, _) = ternary_search_with_index(lower_part, |idx, item| {
                    in_which_part(idx, item).then(Ordering::Greater)
                });
                let (_, upper_idx) = ternary_search_with_index(upper_part, |idx, item| {
                    in_which_part(idx, item).then(Ordering::Less)
                });
                return (low_idx + lower_idx, mid + 1 + upper_idx);
            }
        }
    }

    let span_start_idx = low_idx;
    let span = &data[span_start_idx..high_idx];
    for (sub_idx, item) in span.iter().enumerate() {
        let idx = span_start_idx + sub_idx;
        use core::cmp::Ordering;
        match in_which_part(idx, item) {
            Ordering::Less => low_idx = idx + 1,
            Ordering::Greater => {
                high_idx = idx;
                break;
            }
            Ordering::Equal => (),
        }
    }

    (low_idx, high_idx)
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
        Ok(Self {
            l0_index,
            l1l2_index,
            data,
        })
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

    pub fn rank_ones(&self, idx: u64) -> Option<u64> {
        if idx >= self.len() {
            None
        } else {
            let l0_idx = (idx / size::BITS_PER_L0_BLOCK) as usize;
            let within_l0_idx = idx % size::BITS_PER_L0_BLOCK;
            let l1_idx = (within_l0_idx / size::BITS_PER_L1_BLOCK) as usize;
            let within_l1_idx = within_l0_idx % size::BITS_PER_L1_BLOCK;
            let l2_idx = (within_l1_idx / size::BITS_PER_L2_BLOCK) as usize;
            let within_l2_idx = within_l1_idx % size::BITS_PER_L2_BLOCK;
            debug_assert!(within_l2_idx <= idx);
            let l2_block_start_idx = idx.wrapping_sub(within_l2_idx);
            debug_assert!(l2_block_start_idx <= self.data.len());
            debug_assert!(idx <= self.data.len());
            debug_assert!(l2_block_start_idx <= idx);
            let l0_rank = if l0_idx == 0 {
                0
            } else {
                let get_idx = l0_idx.wrapping_sub(1);
                debug_assert!(self.l0_index.get(get_idx).is_some());
                unsafe { self.l0_index.get_unchecked(get_idx) }.0
            };
            let l1l2_rank = {
                let l1_idx = l1_idx.wrapping_add(l0_idx * size::L1_BLOCKS_PER_L0_BLOCK);
                debug_assert!(self.l1l2_index.get(l1_idx));
                let l1l2_entry = *unsafe { self.l1l2_index.get_unchecked(l1_idx) };
                let l2_index = l1l2_entry.unpack();
                debug_assert!(l2_index.0.get(l2_idx).is_some());
                *unsafe { l2_index.0.get_unchecked(l2_idx as usize) }
            };
            let within_l2_data = unsafe { self.data.get_unchecked_slice(l2_block_start_idx..idx) };
            l0_rank + l1l2_rank as u64 + within_l2_data.count_ones()
        }
    }

    pub fn rank_zeros(&self, idx: u64) -> Option<u64> {
        self.rank_ones(idx)
            .map(|rank_ones| ZeroBits::convert_count(rank_ones, idx))
    }

    fn select<W: OnesOrZeros>(&self, rank: u64) -> Option<u64> {
        if rank >= W::convert_count(self.count_ones(), self.len()) {
            return None;
        }

        let mut skip_bits = 0u64;
        let mut skip_count_ones = 0u64;

        let (_, l0_idx) = ternary_search_with_index(self.l0_index, |idx, l0_entry| {
            // We can ignore that len might be smaller, because we already checked
            // for the end condition above
            let len = ((idx as u64) + 1) * size::BITS_PER_L0_BLOCK;
            let count_ones_at_end_of_block = l0_entry.0;
            let count_at_end_of_block = W::convert_count(count_ones_at_end_of_block, len);
            count_at_end_of_block.cmp(&rank)
        });

        skip_bits += (l0_idx as u64) * size::BITS_PER_L0_BLOCK;
        skip_count_ones += l0_idx
            .checked_sub(1)
            .map_or(0, |prev_block_idx| self.l0_index[prev_block_idx].0);
        let l1l2_part = {
            let lower = l0_idx * size::L1_BLOCKS_PER_L0_BLOCK;
            let upper = core::cmp::min(lower + size::L1_BLOCKS_PER_L0_BLOCK, self.l1l2_index.len());
            debug_assert!(lower < upper);
            self.l1l2_index
                .get(lower..upper)
                .expect("Should be in range or there's a bug")
        };

        let (_, l1_first_index_gt) = ternary_search_with_index(l1l2_part, |idx, l1_entry| {
            let len = (idx as u64) * size::BITS_PER_L1_BLOCK + skip_bits;
            let rank_ones_at_start_of_block = l1_entry.unpack_raw()[0] as u64 + skip_count_ones;
            let rank_at_start_of_block = W::convert_count(rank_ones_at_start_of_block, len);
            rank_at_start_of_block.cmp(&rank)
        });
        debug_assert!(l1_first_index_gt.checked_sub(1).is_some());
        let l1_idx = l1_first_index_gt.wrapping_sub(1);

        skip_bits += (l1_idx as u64) * size::BITS_PER_L1_BLOCK;
        let l2_index = l1l2_part[l1_idx].unpack();
        let l2_first_index_gt = l2_index
            .0
            .iter()
            .enumerate()
            .position(|(idx, rank_ones_within_l0)| {
                let len = (idx as u64) * size::BITS_PER_L2_BLOCK + skip_bits;
                let rank_ones_at_start_of_block = *rank_ones_within_l0 as u64 + skip_count_ones;
                let rank_at_start_of_block = W::convert_count(rank_ones_at_start_of_block, len);
                rank_at_start_of_block > rank
            })
            .unwrap_or(l2_index.0.len());
        debug_assert!(l2_first_index_gt.checked_sub(1).is_some());
        let l2_idx = l1_first_index_gt.wrapping_sub(1);

        skip_bits += (l2_idx as u64) * size::BITS_PER_L2_BLOCK;
        skip_count_ones += l2_index.0[l2_idx] as u64;
        let skip_count = W::convert_count(skip_count_ones, skip_bits);

        let (_skipped, select_in) = self.data.split_at(skip_bits);
        select_in.select::<W>(
            rank.checked_sub(skip_count)
                .expect("Shouldn't have skipped too much"),
        )
    }

    pub fn select_ones(&self, rank: u64) -> Option<u64> {
        self.select::<OneBits>(rank)
    }

    pub fn select_zeros(&self, rank: u64) -> Option<u64> {
        self.select::<ZeroBits>(rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let data = BitsRef::from_bytes(&data[..], 0, 8192 * 2).unwrap();
        let size = IndexSize::for_bits(data);
        let mut l0 = size.l0_vec();
        let mut l1l2 = size.l1l2_vec();
        let index = IndexedBits::from_bits(l0.as_mut_slice(), l1l2.as_mut_slice(), data).unwrap();
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
        let data = BitsRef::from_bytes(&data[..], 0, n_bits).expect("Should have enough bytes");
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

        let gen_sorted_in = |range: core::ops::Range<u64>| {
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
