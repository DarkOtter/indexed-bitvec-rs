use super::{ceil_div, ceil_div_u64, MAX_BITS};
use std::ops::Deref;
use bits_type::Bits;
use ones_or_zeros::{OneBits, ZeroBits, OnesOrZeros};

mod size {
    use super::{ceil_div, ceil_div_u64};

    pub const BITS_PER_L0_BLOCK: u64 = 1 << 32;
    pub const BITS_PER_BLOCK: u64 = 512;

    pub const BYTES_PER_BLOCK: usize = (BITS_PER_BLOCK / 8) as usize;
    pub const BYTES_PER_L0_BLOCK: usize = (BITS_PER_L0_BLOCK / 8) as usize;

    pub fn l0(total_bits: u64) -> usize {
        ceil_div_u64(total_bits, BITS_PER_L0_BLOCK) as usize
    }

    pub fn blocks(total_bits: u64) -> u64 {
        ceil_div_u64(total_bits, BITS_PER_BLOCK)
    }

    pub fn l1l2(total_bits: u64) -> usize {
        ceil_div_u64(blocks(total_bits), 4) as usize
    }

    pub fn rank_index(total_bits: u64) -> usize {
        l0(total_bits) + l1l2(total_bits)
    }

    pub const SAMPLE_LENGTH: u64 = 8192;

    pub fn sample_entries(total_bits: u64) -> usize {
        ceil_div_u64(total_bits, SAMPLE_LENGTH) as usize + l0(total_bits)
    }

    pub fn sample_words(total_bits: u64) -> usize {
        ceil_div(sample_entries(total_bits), 2) + l0(total_bits)
    }

    pub fn total_index_words(total_bits: u64) -> usize {
        l1l2(total_bits) + sample_words(total_bits)
    }

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
    }
}

#[derive(Copy, Clone, Debug)]
struct L1L2Entry(u64);

impl From<u64> for L1L2Entry {
    fn from(i: u64) -> Self {
        L1L2Entry(i)
    }
}

impl From<L1L2Entry> for u64 {
    fn from(packed: L1L2Entry) -> Self {
        packed.0
    }
}

impl L1L2Entry {
    fn pack(base_rank: u32, sub_counts: [u16; 3]) -> Self {
        L1L2Entry(
            ((base_rank as u64) << 32) | ((sub_counts[0] as u64) << 32) |
                ((sub_counts[1] as u64) << 32) | ((sub_counts[2] as u64) << 32),
        )
    }

    fn base_rank(self) -> u64 {
        self.0 >> 32
    }

    fn set_base_rank(self, base_rank: u32) -> Self {
        L1L2Entry(((base_rank as u64) << 32) | self.0 & 0xffffffff)
    }

    fn sub_count_1(self) -> u64 {
        (self.0 >> 22) & 0x3ff
    }

    fn sub_count_2(self) -> u64 {
        (self.0 >> 12) & 0x3ff
    }

    fn sub_count_3(self) -> u64 {
        self.0 & 0x3ff
    }
}

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

            let total = counts.iter().map(|&c| c as u32).sum::<u32>();

            *write_to = L1L2Entry::pack(total, [counts[0], counts[1], counts[2]]).into();
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

    let sub_count_1 = W::convert_count(index_entry.sub_count_1(), size::BITS_PER_BLOCK) as u32;
    if offset_target >= sub_count_1 {
        start_pos += size::BYTES_PER_BLOCK;
        offset_target -= sub_count_1;
        let sub_count_2 = W::convert_count(index_entry.sub_count_2(), size::BITS_PER_BLOCK) as u32;
        if offset_target >= sub_count_2 {
            start_pos += size::BYTES_PER_BLOCK;
            offset_target -= sub_count_2;
            let sub_count_3 = W::convert_count(index_entry.sub_count_3(), size::BITS_PER_BLOCK) as
                u32;
            if offset_target >= sub_count_3 {
                start_pos += size::BYTES_PER_BLOCK;
                offset_target -= sub_count_3;
            }
        }
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

            *write_to = ((high_part << 32) | low_part)
        },
    );

    if (n_samples % 2) != 0 {
        let part_pack = &mut part_packs[0];
        let samples_idx = n_samples - 1;
        let first_sample_offset = (samples_idx as u64 * size::SAMPLE_LENGTH) as u32;
        let high_part = select(first_sample_offset).expect("Should not overflow") as u64;
        *part_pack = (high_part << 32);
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
pub fn index_size_for<T: Deref<Target = [u8]>>(bits: &Bits<T>) -> usize {
    size::total_index_words(bits.used_bits())
}

pub enum IndexingError {
    IncorrectSize,
}

pub fn build_index_for<T: Deref<Target = [u8]>>(
    bits: &Bits<T>,
    into: &mut [u64],
) -> Result<(), IndexingError> {
    let need_size = index_size_for(bits);
    if into.len() != need_size {
        return Err(IndexingError::IncorrectSize);
    } else if bits.used_bits() == 0 {
        debug_assert_eq!(0, need_size);
        return Ok(());
    }

    let bits = bits.clone_ref();
    let l0_size = size::l0(bits.used_bits());
    let (l0_index, index) = into.split_at_mut(l0_size);

    l0_index
        .iter_mut()
        .zip(index.chunks_mut(size::l1l2(size::BITS_PER_L0_BLOCK)).zip(
            bits.chunks_bytes(size::BYTES_PER_L0_BLOCK),
        ))
        .for_each(|(write_l0, (inner_index, l0_chunk))| {
            *write_l0 = build_inner_index(inner_index, l0_chunk) as u64
        });

    let mut total_count = 0u64;
    for l0_index_cell in l0_index.iter_mut() {
        total_count += *l0_index_cell;
        *l0_index_cell = total_count;
    }

    Ok(())
}
