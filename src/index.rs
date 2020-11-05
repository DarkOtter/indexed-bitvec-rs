/*
 * A part of indexed-bitvec-rs, a library implementing bitvectors with fast rank operations.
 *     Copyright (C) 2020  DarkOtter
 *
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */
use crate::bits::BitsRef;
use crate::bits_traits::{OneBits, OnesOrZeros, ZeroBits};
use crate::import::prelude::*;

// TODO: Setup/check testing

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
pub struct L1L2Entry {
    l1_rank: u32,
    l2_entries: PackedL2Entries,
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
struct PackedL2Entries(u32);

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
struct L2Entries([u32; 3]);

impl L0Entry {
    const ZERO: Self = Self(0);
}

impl L1L2Entry {
    const ZERO: Self = Self {
        l1_rank: 0,
        l2_entries: PackedL2Entries::ZERO,
    };
}

impl PackedL2Entries {
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

impl Default for PackedL2Entries {
    fn default() -> Self {
        Self::ZERO
    }
}

impl PackedL2Entries {
    fn pack(items: L2Entries) -> Option<Self> {
        #[inline(always)]
        const fn lossless_low_bits(i: u32, n_bits: u32) -> Option<u32> {
            if n_bits as usize >= size_of::<u32>() * 8 {
                Some(i)
            } else {
                let mask = (1 << n_bits) - 1;
                if (i & !mask) != 0 {
                    None
                } else {
                    Some(i)
                }
            }
        }

        let L2Entries([a, b, c]) = items;
        let a = lossless_low_bits(a, 10)?;
        let b = lossless_low_bits(b, 11)?;
        let c = lossless_low_bits(c, 11)?;
        Some(PackedL2Entries((a << 22) | (b << 11) | (c << 0)))
    }

    fn unpack(self) -> L2Entries {
        #[inline(always)]
        const fn lossy_low_bits(i: u32, n_bits: u32) -> u32 {
            if n_bits as usize >= size_of::<u32>() * 8 {
                i
            } else {
                let mask = (1 << n_bits) - 1;
                i & mask
            }
        }

        let a = lossy_low_bits(self.0 >> 22, 10);
        let b = lossy_low_bits(self.0 >> 11, 11);
        let c = lossy_low_bits(self.0 >> 0, 11);
        L2Entries([a, b, c])
    }
}

impl L1L2Entry {
    pub fn for_bits_with_count_as_l1rank(bits: BitsRef) -> Option<Self> {
        if bits.len() > size::BITS_PER_L1_BLOCK {
            return None;
        }
        debug_assert!(bits.len() <= 4 * size::BITS_PER_L2_BLOCK);
        let mut l2_block_counts = [0u64; 4];
        let chunks = bits
            .chunks(size::BITS_PER_L2_BLOCK)
            .expect("Size should not be zero");
        debug_assert!(chunks.len() <= l2_block_counts.len());
        chunks
            .zip(l2_block_counts.iter_mut())
            .for_each(|(block, count_ones)| *count_ones = block.count_ones());
        l2_block_counts[1] += l2_block_counts[0];
        l2_block_counts[2] += l2_block_counts[1];
        l2_block_counts[3] += l2_block_counts[2];
        debug_assert!(l2_block_counts
            .iter()
            .all(|&x| x <= u32::max_value() as u64));
        Some(L1L2Entry {
            l1_rank: l2_block_counts[3] as u32,
            l2_entries: PackedL2Entries::pack(L2Entries([
                l2_block_counts[0] as u32,
                l2_block_counts[1] as u32,
                l2_block_counts[2] as u32,
            ]))?,
        })
    }

    pub fn rank_at_l2_index(&self, l2_idx: u64) -> Option<u32> {
        if l2_idx >= 4 {
            return None;
        }
        let l2_rank = if l2_idx > 0 {
            self.l2_entries.unpack().0[l2_idx as usize - 1]
        } else {
            0
        };
        Some(self.l1_rank + l2_rank)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct IndexStorage<Upper, Lower> {
    l0: Upper,
    l1l2: Lower,
}

pub type IndexRef<'a> = IndexStorage<&'a [L0Entry], &'a [L1L2Entry]>;

pub struct IndexedBits<Bits, Index> {
    index: Index,
    data: Bits,
}

pub type IndexedBitsRef<'a> = IndexedBits<BitsRef<'a>, IndexRef<'a>>;

fn zip_eq<L, R>(left: L, right: R) -> crate::import::iter::Zip<L, R>
where
    L: crate::import::iter::ExactSizeIterator,
    R: crate::import::iter::ExactSizeIterator,
{
    debug_assert_eq!(
        left.len(),
        right.len(),
        "Iterators are expected to have the same length"
    );
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
        zip_eq(l1l2_index_part.iter_mut(), l1_chunks).for_each(|(entry, data_part)| {
            *entry = L1L2Entry::for_bits_with_count_as_l1rank(data_part)
                .expect("Counts must be in range or it's a bug");
        });
        let mut running_total = 0u64;
        l1l2_index_part.iter_mut().for_each(|entry| {
            assert!(
                running_total <= u32::max_value() as u64,
                "Total rank within l1 should not exceed range of u32 or it's a bug"
            );
            let part_total = entry.l1_rank;
            entry.l1_rank = running_total as u32;
            running_total += part_total as u64;
        });
        running_total
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
    zip_eq(l0_index.iter_mut(), zip_eq(l0_l1l2_chunks, l0_data_chunks)).for_each(
        |(write_count, (l1l2_part, data_part))| {
            *write_count = L0Entry(build_l1(l1l2_part, data_part))
        },
    );

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
        let min = min(a, b);
        let max = max(a, b);
        mid >= min && mid <= max && max - (mid - min + mid) <= 1
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

fn binary_search(mut in_range: Range<u64>, is_greater_or_equal_tgt: impl Fn(u64) -> bool) -> u64 {
    assert!(
        in_range.start <= in_range.end,
        "Range start bound must not be after end bound"
    );

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

impl<'a> IndexRef<'a> {
    pub fn count_ones(&self) -> u64 {
        self.l0.last().map_or(0, |x| x.0)
    }

    fn rank_ones_hint(&self, l2_blocks_from_start: u64) -> Option<u64> {
        let l1_blocks_from_start = (l2_blocks_from_start / size::L2_BLOCKS_PER_L1_BLOCK) as usize;
        let &l1l2_entry = self.l1l2.get(l1_blocks_from_start)?;

        let l2_idx = l2_blocks_from_start % size::L2_BLOCKS_PER_L1_BLOCK;
        let l0_idx = l1_blocks_from_start / size::L1_BLOCKS_PER_L0_BLOCK;

        let l0_rank = if l0_idx == 0 {
            0
        } else {
            let get_idx = l0_idx.wrapping_sub(1);
            debug_assert!(self.l0.get(get_idx).is_some());
            unsafe { self.l0.get_unchecked(get_idx) }.0
        };

        let l1l2_rank = l1l2_entry
            .rank_at_l2_index(l2_idx)
            .expect("L2 index should be in range or it's a bug");

        Some(l0_rank + l1l2_rank as u64)
    }

    fn rank_hint<W: OnesOrZeros>(&self, l2_blocks_from_start: u64) -> Option<u64> {
        self.rank_ones_hint(l2_blocks_from_start).map(|rank_ones| {
            W::convert_count(rank_ones, l2_blocks_from_start * size::BITS_PER_L2_BLOCK)
        })
    }

    fn select_hint<W: OnesOrZeros>(&self, target_rank: u64) -> (u64, u64) {
        let total_l2_blocks = (self.l1l2.len() as u64) * size::L2_BLOCKS_PER_L1_BLOCK;
        let first_possible_l2_block = target_rank / size::BITS_PER_L2_BLOCK;
        let l2_block_with_higher_rank = binary_search(
            first_possible_l2_block..total_l2_blocks,
            |l2_blocks_from_start| {
                self.rank_hint::<W>(l2_blocks_from_start)
                    .expect("If it's not in range that's a bug")
                    > target_rank
            },
        );
        let l2_block_to_search_from = l2_block_with_higher_rank
            .checked_sub(1)
            .expect("The first block must have rank 0, so it can't have a higher rank");
        let l2_block_rank = self
            .rank_hint::<W>(l2_block_to_search_from)
            .expect("If it's not in range that's a bug");
        let idx_to_search_from = l2_block_to_search_from * size::BITS_PER_L2_BLOCK;
        let search_rank = target_rank
            .checked_sub(l2_block_rank)
            .expect("If rank of hint is too high that's a bug");
        (idx_to_search_from, search_rank)
    }
}

impl<'a> IndexedBitsRef<'a> {
    pub(crate) unsafe fn from_existing_index(
        l0_index: &'a [L0Entry],
        l1l2_index: &'a [L1L2Entry],
        data: BitsRef<'a>,
    ) -> Self {
        Self {
            index: IndexStorage {
                l0: l0_index,
                l1l2: l1l2_index,
            },
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
        self.index.count_ones()
    }

    pub fn count_zeros(&self) -> u64 {
        ZeroBits::convert_count(self.count_ones(), self.len())
    }

    pub fn rank_ones(&self, idx: u64) -> Option<u64> {
        if idx >= self.len() {
            None
        } else {
            let l2_blocks_from_start = idx / size::BITS_PER_L2_BLOCK;

            let rank_from_data = {
                let l2_block_start_idx = l2_blocks_from_start * size::BITS_PER_L2_BLOCK;
                let from_l2_block_idx = idx
                    .checked_sub(l2_block_start_idx)
                    .expect("It's impossible for the l2 block to start after the target position");
                let (_data_upto_index_pos, data_from_index_pos) = self
                    .data
                    .split_at(l2_block_start_idx)
                    .expect("If the index is not in range it's a bug");
                data_from_index_pos
                    .rank_ones(from_l2_block_idx)
                    .expect("If the index is not in range it's a bug")
            };

            let rank_from_index = self
                .index
                .rank_ones_hint(l2_blocks_from_start)
                .expect("If the index is not in range it's a bug");

            Some(rank_from_index + rank_from_data)
        }
    }

    pub fn rank_zeros(&self, idx: u64) -> Option<u64> {
        self.rank_ones(idx)
            .map(|rank_ones| ZeroBits::convert_count(rank_ones, idx))
    }

    pub fn select_ones(&self, rank: u64) -> Option<u64> {
        if rank >= self.count_ones() {
            return None;
        }
        let (idx_to_search_from, further_search_rank) = self.index.select_hint::<OneBits>(rank);
        let (_data_to_ignore, data_to_search) = self
            .data
            .split_at(idx_to_search_from)
            .expect("If the index of hint is out of range that's a bug");
        let search_result = data_to_search
            .select_ones(further_search_rank)
            .expect("If the search rank is out of range that's a bug");
        Some(idx_to_search_from + search_result)
    }

    pub fn select_zeros(&self, rank: u64) -> Option<u64> {
        if rank >= self.count_zeros() {
            return None;
        }
        let (idx_to_search_from, further_search_rank) = self.index.select_hint::<ZeroBits>(rank);
        let (_data_to_ignore, data_to_search) = self
            .data
            .split_at(idx_to_search_from)
            .expect("If the index of hint is out of range that's a bug");
        let search_result = data_to_search
            .select_zeros(further_search_rank)
            .expect("If the search rank is out of range that's a bug");
        Some(idx_to_search_from + search_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitsOf;
    use crate::word::Word;

    impl IndexSize {
        fn l0_vec(&self) -> Vec<L0Entry> {
            vec![L0Entry::default(); self.l0_entries()]
        }

        fn l1l2_vec(&self) -> Vec<L1L2Entry> {
            vec![L1L2Entry::default(); self.l1l2_entries()]
        }
    }

    fn lower_u32_of_u64(x: u64) -> u32 {
        (x & (!0u32) as u64) as u32
    }

    #[test]
    fn small_indexed_tests() {
        let n_bits: u64 = (1 << 19) - 1;
        let n_words: usize = ceil_div_u64(n_bits, Word::len()) as usize;
        use oorandom::Rand64;
        let mut rng = Rand64::new(427319723125543870550137410719523151);
        let data = {
            let mut data = vec![Word::zeros(); n_words];
            data.iter_mut()
                .for_each(|cell| *cell = Word::from(lower_u32_of_u64(rng.rand_u64())));
            data
        };
        let data = BitsOf::from(data.as_slice());
        let data = data.split_at(n_bits).unwrap().0;
        let size = IndexSize::for_bits(data);
        let mut l0 = size.l0_vec();
        let mut l1l2 = size.l1l2_vec();
        let index = IndexedBits::from_bits(l0.as_mut_slice(), l1l2.as_mut_slice(), data).unwrap();

        let expected_count_ones = data.count_ones();
        let expected_count_zeros = n_bits - expected_count_ones;
        assert_eq!(index.count_ones() + index.count_zeros(), data.len());
        assert_eq!(expected_count_ones, index.count_ones());
        assert_eq!(expected_count_zeros, index.count_zeros());

        assert_eq!(None, index.rank_ones(n_bits));
        assert_eq!(None, index.rank_zeros(n_bits));

        let mut gen_sorted_in = |range: Range<u64>| {
            let Range { start, end } = range;
            let mut r: Vec<u64> = (0..1000).map(|_| rng.rand_range(start..end)).collect();
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
