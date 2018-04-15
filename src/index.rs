use super::{mult_64, div_64, mod_64, ceil_div_64};
use words::{WordData, BitData};
use bits64::Bits64;
use bitvec64::BitVec64;

#[inline]
fn ceil_div(n: usize, d: usize) -> usize {
    (n / d) + if n % d > 0 { 1 } else { 0 }
}

mod size {
    use super::ceil_div;

    pub const BITS_PER_L0_BLOCK: usize = 1 << 32;
    pub const BITS_PER_BLOCK: usize = 512;

    pub fn l0(total_bits: usize) -> usize {
        if total_bits <= BITS_PER_L0_BLOCK {
            0
        } else {
            (total_bits - 1) >> 32
        }
    }

    pub fn blocks(total_bits: usize) -> usize {
        if total_bits == 0 {
            0
        } else {
            ((total_bits - 1) >> 9) + 1
        }
    }

    pub fn l1l2(total_bits: usize) -> usize {
        if total_bits < 512 {
            0
        } else {
            ceil_div(blocks(total_bits), 1 << 2)
        }
    }

    pub fn rank_index(total_bits: usize) -> usize {
        l0(total_bits) + l1l2(total_bits)
    }
}

fn read_l1l2<I: WordData>(index: &I, l0_size: usize, block_idx: usize) -> u32 {
    if block_idx == 0 {
        return 0;
    }
    let read_idx = l0_size + block_idx >> 2;
    let l2_offset = block_idx & 3;
    let data = index.get_word(read_idx);
    let parts: [u32; 4] = [
        ((data >> 32) & 0xffffffff) as u32,
        ((data >> 22) & 0x3ff) as u32,
        ((data >> 12) & 0x3ff) as u32,
        ((data >> 2) & 0x3ff) as u32,
    ];
    if l2_offset < 2 {
        if l2_offset == 0 {
            parts[0]
        } else {
            parts[0] + parts[1]
        }
    } else {
        let half = parts[0] + parts[1];
        if l2_offset == 2 {
            half + parts[2]
        } else {
            half + parts[2] + parts[3]
        }
    }
}

fn unsafe_rank<I, D>(index: &I, data: &D, idx_bits: usize) -> usize
where
    I: WordData,
    D: WordData,
{
    let total_words = data.len_words();
    let l0_size = size::l0(total_words);

    let block_index = idx_bits >> 9;

    let word_index = idx_bits >> 6;
    let index_in_word = idx_bits & 0x3f;
    let block_start_words = block_index << 3;

    let l0_count = {
        if idx_bits < 1 << 32 {
            0
        } else {
            index.get_word((idx_bits >> 32) - 1)
        }
    };

    let l1l2_count = read_l1l2(index, l0_size, block_index);

    let subword_count = Bits64::from(data.get_word(word_index)).rank(index_in_word);

    let whole_words_count = {
        let mut count = 0;
        for i in block_start_words..word_index {
            count += data.count_ones_word(i);
        }
        count
    };

    l0_count as usize + (l1l2_count + subword_count + whole_words_count) as usize
}

fn check_index_size<I, D>(index: &I, data: &D)
where
    I: WordData,
    D: BitData,
{
    let expected_size = size::rank_index(data.len_bits());
    if expected_size != index.len_words() {
        panic!("Index length mismatches data");
    }
}

#[inline]
fn rank<I, D>(index: &I, data: &D, idx_bits: usize) -> usize
where
    I: WordData,
    D: BitData,
{
    check_index_size(index, data);

    if idx_bits >= data.len_bits() {
        panic!("Index out of bounds");
    }

    unsafe_rank(index, data, idx_bits)
}

#[inline]
fn rank_zeros<I, D>(index: &I, data: &D, idx_bits: usize) -> usize
where
    I: WordData,
    D: BitData,
{
    idx_bits - rank(index, data, idx_bits)
}

fn count_ones<I, D>(index: &I, data: &D) -> usize
where
    I: WordData,
    D: BitData,
{
    check_index_size(index, data);

    let last_idx = if data.len_bits() == 0 {
        return 0;
    } else {
        data.len_bits() - 1
    };

    let last_word = div_64(last_idx);
    let count_in_last_word = {
        let last_word = Bits64::from(data.get_word(last_word));
        let idx_in_last_word = mod_64(last_idx) + 1;
        if idx_in_last_word < 64 {
            last_word.rank(idx_in_last_word)
        } else {
            last_word.count_ones()
        }
    };
    let pre_rank = {
        let last_word_start = mult_64(last_word);
        unsafe_rank(index, data, last_word_start)
    };
    pre_rank + count_in_last_word as usize
}

#[inline]
fn count_zeros<I, D>(index: &I, data: &D) -> usize
where
    I: WordData,
    D: BitData,
{
    data.len_bits() - count_ones(index, data)
}

pub struct RankIndex(BitVec64);

struct WordPopcountIter<'a, D: BitData + 'a> {
    data: &'a D,
    idx: usize,
}

impl<'a, D: BitData + 'a> WordPopcountIter<'a, D> {
    fn from(data: &'a D) -> Self {
        WordPopcountIter { data, idx: 0 }
    }
}

impl<'a, D: BitData + 'a> Iterator for WordPopcountIter<'a, D> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let bits = self.data.len_bits();
        let whole_words = div_64(bits);

        if self.idx < whole_words {
            let res = self.data.count_ones_word(self.idx);
            self.idx += 1;
            Some(res)
        } else if self.idx > whole_words {
            None
        } else {
            let partial_bits = mod_64(bits);
            if partial_bits > 0 {
                let res = Bits64::from(self.data.get_word(self.idx)).rank(partial_bits);
                self.idx += 1;
                Some(res)
            } else {
                self.idx += 1;
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = ceil_div_64(self.data.len_bits()).saturating_sub(self.idx);
        (remaining, Some(remaining))
    }
}

struct BlockPopcountIter<'a, D: BitData + 'a>(WordPopcountIter<'a, D>);

impl<'a, D: BitData + 'a> BlockPopcountIter<'a, D> {
    fn from(data: &'a D) -> Self {
        BlockPopcountIter(WordPopcountIter::from(data))
    }
}

impl<'a, D: BitData + 'a> Iterator for BlockPopcountIter<'a, D> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        let mut total = match self.0.next() {
            None => return None,
            Some(x) => x,
        };

        for _ in 0..7 {
            match self.0.next() {
                None => break,
                Some(x) => total += x,
            }
        }

        Some(total)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (min_remain, max_remain) = self.0.size_hint();
        let min_remain = min_remain / 8;
        let max_remain = max_remain.map(|x| ceil_div(x, 8));
        (min_remain, max_remain)
    }
}

fn pack_l1l2(counts: [u32; 4]) -> Bits64 {
    debug_assert!(counts[3] - counts[2] <= 512);
    debug_assert!(counts[2] - counts[1] <= 512);
    debug_assert!(counts[1] - counts[0] <= 512);

    let part_0 = (counts[0] as u64) << 32;
    let part_1 = ((counts[1] - counts[0]) << 22) as u64;
    let part_2 = ((counts[2] - counts[1]) << 12) as u64;
    let part_3 = ((counts[3] - counts[2]) << 2) as u64;

    Bits64::from(part_0 + part_1 + part_2 + part_3)
}

impl RankIndex {
    pub fn index<D: BitData>(data: &D) -> Self {
        // TODO: Add total set count to end of L0 index
        // TODO: Build without using extra space
        let total_bits = data.len_bits();
        if total_bits < 512 {
            debug_assert!(size::l0(total_bits) == 0);
            debug_assert!(size::l1l2(total_bits) == 0);
            return RankIndex(BitVec64::from_data(Vec::with_capacity(0)));
        }

        let block_popcounts: Vec<u32> = BlockPopcountIter::from(data).collect();

        let l0_size = size::l0(total_bits);
        let l1l2_size = size::l1l2(total_bits);

        debug_assert!(l1l2_size == ceil_div(block_popcounts.len(), 4));

        let mut res = Vec::with_capacity(l0_size + l1l2_size);
        for _ in 0..l0_size {
            res.push(Bits64::ZEROS)
        }
        let mut l0_idx = 0;

        let mut l0_rank = 0u64;
        for l0_chunk in block_popcounts.as_slice().chunks(
            size::BITS_PER_L0_BLOCK /
                size::BITS_PER_BLOCK,
        )
        {
            if l0_idx > 0 {
                res[l0_idx - 1] = Bits64::from(l0_rank);
            }
            l0_idx += 1;

            let mut lower_rank = 0u32;
            let (whole_chunks, partial_l1l2_chunk) = l0_chunk.split_at(l0_chunk.len() & !3);

            for l1l2_chunk in whole_chunks.chunks(4) {
                let mut ranks = [lower_rank; 4];
                for i in 0..3 {
                    ranks[i + 1] = ranks[i] + l1l2_chunk[i]
                }
                lower_rank = ranks[3] + l1l2_chunk[3];
                res.push(pack_l1l2(ranks));
            }

            // Final (possibly partial) set of 4 blocks
            if partial_l1l2_chunk.len() > 0 {
                let mut ranks = [lower_rank; 4];
                // Slow path
                for (i, c) in partial_l1l2_chunk.iter().cloned().enumerate() {
                    ranks[i + 1] = ranks[i] + c;
                }
                let highest_rank = ranks[partial_l1l2_chunk.len()];
                for i in (partial_l1l2_chunk.len() + 1)..4 {
                    ranks[i] = highest_rank
                }
                lower_rank = ranks[3];
                res.push(pack_l1l2(ranks));
            }

            l0_rank += lower_rank as u64;
        }

        debug_assert!(res.len() == l0_size + l1l2_size);
        debug_assert!(l0_idx == l0_size + 1);

        RankIndex(BitVec64::from_data(res))
    }

    #[inline]
    pub fn rank<D: BitData>(&self, data: &D, idx_bits: usize) -> usize {
        rank(&self.0, data, idx_bits)
    }

    #[inline]
    pub fn rank_zeros<D: BitData>(&self, data: &D, idx_bits: usize) -> usize {
        rank_zeros(&self.0, data, idx_bits)
    }

    #[inline]
    pub fn count_ones<D: BitData>(&self, data: &D) -> usize {
        count_ones(&self.0, data)
    }

    #[inline]
    pub fn count_zeros<D: BitData>(&self, data: &D) -> usize {
        count_zeros(&self.0, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    quickcheck! {
        fn test_rank_and_count_fns(x: BitVec64) -> () {
            let index = RankIndex::index(&x);

            let mut manual_rank = 0;
            for idx in 0..x.len_bits() {
                assert_eq!(index.rank(&x, idx), manual_rank);
                assert_eq!(index.rank_zeros(&x, idx), idx - manual_rank);
                if x.get_bit(idx) { manual_rank += 1 }
            }

            assert_eq!(index.count_ones(&x), manual_rank);
            assert_eq!(index.count_zeros(&x), x.len_bits() - manual_rank);
        }
    }
}
