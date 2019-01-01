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
//! Functions to run bit operations on bits stored as bytes (MSB first).
use crate::ones_or_zeros::OnesOrZeros;
use crate::word::{select_ones_u16, Word};

#[inline]
pub(crate) fn get_unchecked(data: &[u8], idx_bits: u64) -> bool {
    let byte_idx = (idx_bits / 8) as usize;
    let idx_in_byte = (idx_bits % 8) as usize;

    let byte = data[byte_idx];
    let mask = 0x80 >> idx_in_byte;
    (byte & mask) != 0
}

#[cfg(test)]
fn get(data: &[u8], idx_bits: u64) -> Option<bool> {
    let byte_idx = idx_bits / 8;
    if byte_idx >= data.len() as u64 {
        None
    } else {
        Some(get_unchecked(data, idx_bits))
    }
}

pub(crate) fn set_unchecked(data: &mut [u8], idx_bits: u64, to: bool) {
    let byte_idx = (idx_bits / 8) as usize;
    let idx_in_byte = (idx_bits % 8) as usize;

    let mask = 0x80 >> idx_in_byte;

    if to {
        data[byte_idx] |= mask
    } else {
        data[byte_idx] &= !mask
    }
}

fn bytes_as_u64s(data: &[u8]) -> Result<(&[u8], &[u64], &[u8]), &[u8]> {
    use core::mem::{align_of, size_of};

    if data.len() < size_of::<u64>() || data.len() < align_of::<u64>() {
        return Err(data);
    }

    // For actually casting we must match alignment
    let alignment_offset = {
        let need_alignment = align_of::<u64>();
        let rem = (data.as_ptr() as usize) % need_alignment;
        if rem > 0 { need_alignment - rem } else { 0 }
    };

    let (pre_partial, data) = data.split_at(alignment_offset);

    let n_whole_words = data.len() / size_of::<u64>();

    let (data, post_partial) = data.split_at(n_whole_words * size_of::<u64>());

    let data: &[u64] = unsafe {
        use core::slice::from_raw_parts;
        let ptr = data.as_ptr() as *const u64;
        from_raw_parts(ptr, n_whole_words)
    };

    Ok((pre_partial, data, post_partial))
}

fn count_ones_with<T: Copy, F: Fn(T) -> u32>(data: &[T], count_ones: F) -> u64 {
    data.iter().map(|&x| count_ones(x) as u64).sum::<u64>()
}

fn count_ones_bytes_slow(data: &[u8]) -> u64 {
    count_ones_with(data, <u8>::count_ones)
}

fn count_ones_words(data: &[u64]) -> u64 {
    count_ones_with(data, <u64>::count_ones)
}

pub fn count_ones(data: &[u8]) -> u64 {
    match bytes_as_u64s(data) {
        Err(data) => count_ones_bytes_slow(data),
        Ok((pre_partial, data, post_partial)) => {
            count_ones_bytes_slow(pre_partial) + count_ones_words(data) +
                count_ones_bytes_slow(post_partial)
        }
    }
}

pub fn rank_ones(data: &[u8], idx_bits: u64) -> Option<u64> {
    let full_bytes = idx_bits / 8;
    if full_bytes >= data.len() as u64 {
        return None;
    }

    let full_bytes = full_bytes as usize;
    let rem = idx_bits % 8;
    let full_bytes_count = count_ones(&data[..full_bytes]);
    let rem_count = (!((!0u8) >> rem) & data[full_bytes]).count_ones();
    Some(full_bytes_count + rem_count as u64)
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

pub fn select<W: OnesOrZeros>(data: &[u8], target_rank: u64) -> Option<u64> {
    let split_res = bytes_as_u64s(data);

    let pre_partial = match split_res {
        Err(data) => data,
        Ok((pre, _, _)) => pre,
    };

    let pre_partial_count = match select_by_bytes::<W>(pre_partial, target_rank) {
        Ok(res) => return Some(res),
        Err(count) => count,
    };

    let (data, post_partial) = match split_res {
        Err(_) => return None,
        Ok((_, data, post)) => (data, post),
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
    use std::vec::Vec;
    use crate::ones_or_zeros::{OneBits, ZeroBits};

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
        fn test_set_unchecked(bools: Vec<bool>) -> bool {
            let mut data = vec![0; (bools.len() + 7) / 8].into_boxed_slice();
            bools.iter().cloned().enumerate().for_each(
                |(idx, boolean)| set_unchecked(&mut data[..], idx as u64, boolean));
            bools.iter().cloned().enumerate().all(
                |(idx, boolean)| Some(boolean) == get(&data[..], idx as u64))
        }

        fn test_count(data: Vec<u8>) -> bool {
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
    fn test_rank() {
        let data = random_data();

        let mut expected_rank_ones = 0u64;
        let mut expected_rank_zeros = 0u64;

        for (byte_idx, byte) in data.iter().cloned().enumerate() {
            let byte_idx = byte_idx as u64;
            let bits = bits_of_byte(byte);
            for (bit_idx, bit) in bits.iter().cloned().enumerate() {
                let bit_idx = bit_idx as u64;
                let rank_ones = rank_ones(&data, byte_idx * 8 + bit_idx);
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
    }

    fn do_test_select<W: OnesOrZeros>(data: &Vec<u8>) {
        let total_count = W::convert_count(count_ones(&data), data.len() as u64 * 8) as usize;
        for i in 0..total_count {
            let i = i as u64;
            let r = select::<W>(&data, i).expect("Already checked in-bounds");
            let rank = W::convert_count(rank_ones(&data, r).unwrap(), r);
            assert_eq!(i, rank);
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
