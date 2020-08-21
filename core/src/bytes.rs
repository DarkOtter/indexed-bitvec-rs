/*
   Copyright 2020 DarkOtter

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
//! Functions to run bit operations on bits stored as bytes (MSB first).
use crate::ones_or_zeros::OnesOrZeros;
use crate::word::{select_ones_u16, Word};

#[inline]
pub unsafe fn get_unchecked(data: &[u8], idx_bits: u64) -> bool {
    let byte_idx = (idx_bits / 8) as usize;
    let idx_in_byte = (idx_bits % 8) as usize;

    debug_assert!(data.get(byte_idx).is_some());
    let byte = data.get_unchecked(byte_idx);
    let mask = 0x80 >> idx_in_byte;
    (byte & mask) != 0
}

#[cfg(test)]
pub fn get(data: &[u8], idx_bits: u64) -> Option<bool> {
    let byte_idx = idx_bits / 8;
    if byte_idx >= data.len() as u64 {
        None
    } else {
        Some(unsafe { get_unchecked(data, idx_bits) })
    }
}

fn bytes_as_u64s(data: &[u8]) -> (&[u8], &[u64], &[u8]) {
    const WORD_ALIGNMENT: usize = core::mem::align_of::<u64>();
    const WORD_SIZE: usize = core::mem::size_of::<u64>();

    let total_bytes = data.len();
    if total_bytes < WORD_ALIGNMENT {
        return (data, &[], &[]);
    }

    // For actually casting we must match alignment
    let words_start = {
        let rem = (data.as_ptr() as usize) % WORD_ALIGNMENT;
        if rem != 0 {
            WORD_ALIGNMENT - rem
        } else {
            0
        }
    };

    debug_assert!(total_bytes > words_start);
    let n_words = total_bytes.wrapping_sub(words_start) / WORD_SIZE;
    let words_end = words_start + n_words * WORD_SIZE;

    debug_assert!(
        words_start <= data.len() && words_end <= data.len() && total_bytes == data.len()
    );
    let first_part = unsafe { data.get_unchecked(0..words_start) };
    let words_part = unsafe { data.get_unchecked(words_start..words_end) };
    let last_part = unsafe { data.get_unchecked(words_end..total_bytes) };

    let words: &[u64] = unsafe {
        use core::slice::from_raw_parts;
        debug_assert_eq!((words_part.as_ptr() as usize) % WORD_ALIGNMENT, 0);
        let ptr = words_part.as_ptr() as *const u64;
        from_raw_parts(ptr, n_words)
    };

    (first_part, words, last_part)
}

const MIN_SIZE_TO_SPLIT_WORDS: usize = 16 * core::mem::size_of::<u64>();

fn count_ones_with<T: Copy, F: Fn(T) -> u32>(data: &[T], count_ones: F) -> u64 {
    data.iter().map(|&x| count_ones(x) as u64).sum::<u64>()
}

fn count_ones_by_bytes(data: &[u8]) -> u64 {
    count_ones_with(data, <u8>::count_ones)
}

fn count_ones_by_words(data: &[u64]) -> u64 {
    count_ones_with(data, <u64>::count_ones)
}

pub fn count_ones(data: &[u8]) -> u64 {
    if data.len() >= MIN_SIZE_TO_SPLIT_WORDS {
        let (pre_partial, words, post_partial) = bytes_as_u64s(data);
        count_ones_by_bytes(pre_partial)
            + count_ones_by_words(words)
            + count_ones_by_bytes(post_partial)
    } else {
        count_ones_by_bytes(data)
    }
}

pub unsafe fn count_ones_upto_unchecked(data: &[u8], idx_bits: u64) -> u64 {
    let full_bytes = (idx_bits / 8) as usize;
    let mut count_so_far = count_ones(data.get_unchecked(..full_bytes));

    let rem = idx_bits % 8;
    if rem != 0 {
        let partial_byte = data.get_unchecked(full_bytes);
        count_so_far += (!((!0u8) >> rem) & partial_byte).count_ones() as u64;
    }
    count_so_far
}

pub fn count_ones_upto(data: &[u8], idx_bits: u64) -> Option<u64> {
    if indx_bits <= (data.len() as u64) * 8 {
        Some(unsafe { count_ones_upto_unchecked(data, idx_bits) })
    } else {
        None
    }
}

/// Select a bit by rank within bytes one byte at a time, or return the total count.
fn select_by_bytes<W: OnesOrZeros>(data: &[u8], target_rank: u64) -> Result<u64, u64> {
    let mut running_rank = 0u64;
    let mut running_index = 0u64;

    for &byte in data.iter() {
        let count = W::convert_count(byte.count_ones() as u64, 8);
        if running_rank + count > target_rank {
            let select_in = if W::is_ones() {
                byte as u16
            } else {
                (!byte) as u16
            };
            let selected = select_ones_u16(select_in, (target_rank - running_rank) as u32);
            let answer = selected as u64 - 8 + running_index;
            return Ok(answer);
        }
        running_rank += count;
        running_index += 8;
    }

    Err(running_rank)
}

pub(crate) fn select<W: OnesOrZeros>(data: &[u8], target_rank: u64) -> Option<u64> {
    if data.len() < MIN_SIZE_TO_SPLIT_WORDS {
        return select_by_bytes::<W>(data, target_rank).ok();
    }

    let (pre_partial, data, post_partial) = bytes_as_u64s(data);
    let pre_partial_count = match select_by_bytes::<W>(pre_partial, target_rank) {
        Ok(res) => return Some(res),
        Err(count) => count,
    };

    let mut running_rank = pre_partial_count;
    let mut running_index = pre_partial.len() as u64 * 8;

    for &word in data.iter() {
        let count = W::convert_count(word.count_ones() as u64, 64);
        if running_rank + count > target_rank {
            let answer = Word::from(u64::from_be(word))
                .select::<W>((target_rank - running_rank) as u32)
                .map(|sub_res| running_index + sub_res as u64);
            return answer;
        }
        running_rank += count;
        running_index += 64;
    }

    select_by_bytes::<W>(post_partial, target_rank - running_rank)
        .ok()
        .map(|sub_res| running_index + sub_res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ones_or_zeros::{OneBits, ZeroBits};
    use std::vec::Vec;

    #[test]
    fn test_get() {
        let pattern_a = [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];
        let bytes_a = &pattern_a[..];
        for i in 0..(bytes_a.len() * 8) {
            assert_eq!(
                i / 8 == i % 8,
                get(bytes_a, i as u64).unwrap(),
                "Differed at position {}",
                i
            )
        }
        assert_eq!(None, get(bytes_a, bytes_a.len() as u64 * 8));

        let pattern_b = [0xff, 0xc0];
        let bytes_b = &pattern_b[..];
        for i in 0..10 {
            assert_eq!(Some(true), get(bytes_b, i), "Differed at position {}", i)
        }
        for i in 10..16 {
            assert_eq!(Some(false), get(bytes_b, i), "Differed at position {}", i)
        }
    }

    quickcheck! {
        fn test_count_ones(data: Vec<u8>) -> bool {
            let mut expected_count_ones = 0u64;
            let mut expected_count_zeros = 0u64;

            for byte in data.iter().cloned() {
                expected_count_ones += byte.count_ones() as u64;
                expected_count_zeros += byte.count_zeros() as u64;
            }

            let count_ones = count_ones(&data);
            let count_zeros = ZeroBits::convert_count(count_ones, data.len() as u64 * 8);

            count_ones == expected_count_ones
                && count_zeros == expected_count_zeros
        }
    }

    fn bits_of_byte(b: u8) -> [bool; 8] {
        let mut res = [false; 8];
        for (i, r) in res.iter_mut().enumerate() {
            *r = (b & (1 << (7 - i))) > 0
        }
        res
    }

    fn random_data() -> Vec<u8> {
        use rand::{thread_rng, RngCore};
        let mut res = vec![0u8; 3247];
        thread_rng().fill_bytes(&mut res);
        res
    }

    #[test]
    fn test_count_ones_upto() {
        let data = random_data();

        let mut expected_rank_ones = 0u64;
        let mut expected_rank_zeros = 0u64;

        for (byte_idx, byte) in data.iter().cloned().enumerate() {
            let byte_idx = byte_idx as u64;
            let bits = bits_of_byte(byte);
            for (bit_idx, bit) in bits.iter().cloned().enumerate() {
                let bit_idx = bit_idx as u64;
                let rank_ones = count_ones_upto(&data, byte_idx * 8 + bit_idx);
                assert_eq!(Some(expected_rank_ones), rank_ones);
                let rank_zeros =
                    ZeroBits::convert_count(rank_ones.unwrap(), byte_idx * 8 + bit_idx);
                assert_eq!(expected_rank_zeros, rank_zeros);

                if bit {
                    expected_rank_ones += 1;
                } else {
                    expected_rank_zeros += 1;
                }
            }
        }

        let len_bits = data.len() as u64 * 8;
        let rank_ones = count_ones_upto(&data, len_bits);
        assert_eq!(Some(expected_rank_ones), rank_ones);
        let rank_zeros = ZeroBits::convert_count(rank_ones.unwrap(), len_bits);
        assert_eq!(expected_rank_zeros, rank_zeros);
    }

    fn do_test_select<W: OnesOrZeros>(data: &Vec<u8>) {
        let total_count = W::convert_count(count_ones(&data), data.len() as u64 * 8) as usize;
        for i in 0..total_count {
            let i = i as u64;
            let r = select::<W>(&data, i).expect("Already checked in-bounds");
            let rank = W::convert_count(count_ones_upto(&data, r).unwrap(), r);
            assert_eq!(i, rank);
            assert_eq!(Some(W::is_ones()), get(&data, r));
        }
        assert_eq!(None, select::<W>(&data, total_count as u64));
    }

    #[test]
    fn test_select() {
        let data = random_data();

        do_test_select::<OneBits>(&data);
        do_test_select::<ZeroBits>(&data);
    }
}
