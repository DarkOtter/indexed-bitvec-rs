use super::{div_64, mod_64, mult_64, ceil_div, MAX_BITS, MAX_BITS_IN_BYTES};
use std::mem::size_of;
use std::cmp::min;
use word::{select_ones_u16, Word};
use byteorder::{BigEndian, ByteOrder};
use ones_or_zeros::OnesOrZeros;

pub(crate) fn get_unchecked(data: &[u8], idx_bits: usize) -> bool {
    let byte_idx = idx_bits >> 3;
    let idx_in_byte = idx_bits & 7;

    let byte = data[byte_idx];
    let mask = 0x80 >> idx_in_byte;
    (byte & mask) != 0
}

pub fn get(data: &[u8], idx_bits: usize) -> Option<bool> {
    if idx_bits >> 3 >= data.len() {
        None
    } else {
        Some(get_unchecked(data, idx_bits))
    }
}

/*
const MAX_WORD_IDX: usize = (<usize>::max_value() - size_of::<u64>()) / size_of::<u64>();

pub fn read_word(data: &[u8], idx: usize) -> Option<Word> {
    if idx > MAX_WORD_IDX {
        // Avoid overflow
        return None;
    };
    let idx_bytes = idx * size_of::<u64>();
    let end_idx_bytes = idx_bytes + size_of::<u64>();
    if end_idx_bytes > data.len() {
        read_word_last_word_case(data, idx_bytes)
    } else {
        Some(BigEndian::read_u64(&data[idx_bytes..end_idx_bytes]).into())
    }
}

#[cold]
fn read_word_last_word_case(data: &[u8], idx_bytes: usize) -> Option<Word> {
    if idx_bytes >= data.len() {
        return None;
    }
    debug_assert!((data.len() - idx_bytes) < size_of::<u64>());
    read_partial_word(&data[idx_bytes..])
}


fn read_partial_word(data: &[u8]) -> Option<Word> {
    if data.len() >= size_of::<u64>() {
        return None;
    } else if data.len() == 0 {
        return Some(Word::ZEROS);
    }

    let mut res = [0u8; size_of::<u64>()];
    res[0..data.len()].copy_from_slice(data);
    Some(BigEndian::read_u64(&res).into())
}

pub fn len_words(data: &[u8]) -> usize {
    ceil_div(data.len(), size_of::<u64>())
}
*/

fn bytes_as_u64s(data: &[u8]) -> Result<(&[u8], &[u64], &[u8]), &[u8]> {
    use std::mem::{size_of, align_of};

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
        use std::slice::from_raw_parts;
        let ptr = data.as_ptr() as *const u64;
        from_raw_parts(ptr, n_whole_words)
    };

    Ok((pre_partial, data, post_partial))
}

/// Unsafe because it could overflow the count
fn count_ones_unsafe<T: Copy, F: Fn(T) -> u32>(count_ones: F, data: &[T]) -> u64 {
    data.iter().map(|&x| count_ones(x) as u64).sum::<u64>()
}

/// Unsafe because it could overflow the count
fn count_ones_base_unsafe(data: &[u8]) -> u64 {
    count_ones_unsafe(<u8>::count_ones, data)
}

pub fn count_ones(data: &[u8]) -> Option<u64> {
    if data.len() > MAX_BITS_IN_BYTES {
        // Avoid overflow
        return None;
    }

    match bytes_as_u64s(data) {
        Err(data) => Some(count_ones_base_unsafe(data)),
        Ok((pre_partial, data, post_partial)) => Some(
            count_ones_base_unsafe(pre_partial) + count_ones_unsafe(<u64>::count_ones, data) +
                count_ones_base_unsafe(post_partial),
        ),
    }
}

pub fn count<W: OnesOrZeros>(data: &[u8]) -> Option<u64> {
    count_ones(data).map(|count_ones| {
        W::convert_count(count_ones, data.len() as u64 * 8)
    })
}

pub fn rank_ones(data: &[u8], idx_bits: u64) -> Option<u64> {
    if idx_bits > MAX_BITS {
        // Avoid overflow
        return None;
    }

    let full_bytes = (idx_bits / 8) as usize;

    if full_bytes >= data.len() {
        return None;
    }

    let rem = idx_bits % 8;
    let full_bytes_count =
        count_ones(&data[..full_bytes]).expect("Already checked for too-many-bits");
    let rem_count = (!((!0u8) >> rem) & data[full_bytes]).count_ones();
    Some(full_bytes_count + rem_count as u64)
}

pub fn rank<W: OnesOrZeros>(data: &[u8], idx_bits: u64) -> Option<u64> {
    rank_ones(data, idx_bits).map(|count_ones| W::convert_count(count_ones, idx_bits))
}

// If it returns Err it returns the total count for the data
fn select_base<W: OnesOrZeros>(data: &[u8], target_rank: u64) -> Result<u64, u64> {
    let mut running_rank = 0u64;
    let mut running_index = 0u64;

    for &byte in data.iter() {
        let count = W::convert_count(byte.count_ones() as u64, 8);
        if running_rank + count > target_rank {
            // TODO: This uses select ones...
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
    let parts = bytes_as_u64s(data);

    let (pre_partial, data, post_partial) = match bytes_as_u64s(data) {
        Err(data) => return select_base::<W>(data, target_rank).ok(),
        Ok(x) => x,
    };

    let pre_partial_count = match select_base::<W>(pre_partial, target_rank) {
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

    select_base::<W>(post_partial, target_rank - running_rank)
        .ok()
        .map(|sub_res| running_index + sub_res)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck;
    use ones_or_zeros::{OneBits, ZeroBits};

    #[test]
    fn test_get() {
        let pattern_a = [0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];
        let bytes_a = &pattern_a[..];
        for i in 0..(bytes_a.len() * 8) {
            assert_eq!(
                i / 8 == i % 8,
                get(bytes_a, i).unwrap(),
                "Differed at position {}",
                i
            )
        }
        assert_eq!(None, get(bytes_a, bytes_a.len() * 8));

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
        fn test_count(data: Vec<u8>) -> bool {
            let mut count_ones = 0u64;
            let mut count_zeros = 0u64;

            for byte in data.iter().cloned() {
                count_ones += byte.count_ones() as u64;
                count_zeros += byte.count_zeros() as u64;
            }

            count::<OneBits>(&data) == Some(count_ones)
                && count::<ZeroBits>(&data) == Some(count_zeros)
        }
    }

    fn bits_of_byte(b: u8) -> [bool; 8] {
        let mut res = [false; 8];
        for (i, r) in res.iter_mut().enumerate() {
            *r = (b & (1 << (7 - i))) > 0
        }
        res
    }

    #[test]
    fn test_rank() {
        use rand::thread_rng;
        let mut gen = quickcheck::StdGen::new(thread_rng(), 1024);
        let data = <Vec<u8> as quickcheck::Arbitrary>::arbitrary(&mut gen);

        let mut rank_ones = 0u64;
        let mut rank_zeros = 0u64;

        for (byte_idx, byte) in data.iter().cloned().enumerate() {
            let byte_idx = byte_idx as u64;
            let bits = bits_of_byte(byte);
            for (bit_idx, bit) in bits.iter().cloned().enumerate() {
                let bit_idx = bit_idx as u64;
                assert_eq!(
                    Some(rank_ones),
                    rank::<OneBits>(&data, byte_idx * 8 + bit_idx)
                );
                assert_eq!(
                    Some(rank_zeros),
                    rank::<ZeroBits>(&data, byte_idx * 8 + bit_idx)
                );

                if bit {
                    rank_ones += 1;
                } else {
                    rank_zeros += 1;
                }
            }
        }
    }

    fn do_test_select<W: OnesOrZeros>(data: &Vec<u8>) {
        let total_count = count::<W>(&data).unwrap() as usize;
        for i in 0..total_count {
            let i = i as u64;
            let r = select::<W>(&data, i).expect("Already checked in-bounds");
            assert_eq!(Some(i), rank::<W>(&data, r));
        }
        assert_eq!(None, select::<W>(&data, total_count as u64));
    }

    #[test]
    fn test_select() {
        use rand::thread_rng;
        let mut gen = quickcheck::StdGen::new(thread_rng(), 1024);
        let data = <Vec<u8> as quickcheck::Arbitrary>::arbitrary(&mut gen);

        do_test_select::<OneBits>(&data);
        do_test_select::<ZeroBits>(&data);
    }

    /*

    #[test]
    fn test_read_word_basic() {
        let pattern = [0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 5];
        let data = &pattern[..];
        assert_eq!(data.len(), 15);
        assert_eq!(u64::from(read_word(data, 0).unwrap()), 5u64);
        assert_eq!(u64::from(read_word(data, 1).unwrap()), 5u64 * 256);
    }

    quickcheck! {
        fn test_read_word(bits: OwnedBits) -> () {
            for word_idx in 0..ceil_div(bits.used_bits(), 64) {
                let word = read_word(bits.bytes(), word_idx).unwrap();
                for i in 0..64 {
                    let bit_idx = word_idx * 64 + i;
                    match bits.get(bit_idx) {
                        None => {
                            assert!(bit_idx >= bits.used_bits());
                            if bit_idx / 8 >= bits.bytes().len() {
                            
                            assert_eq!(word.get(i), Some(false), "Failed padding word (word {}, bit {})", word_idx, i);
                            }
                        },
                        original_bit => {
                            assert_eq!(word.get(i), original_bit, "Failed preserving bit (word {}, bit {})", word_idx, i);
                        }
                    }
                }
            }
        }

        fn test_count(bits: OwnedBits) -> () {
            let count_ones = count::<OneBits>(bits.bytes());
        }
    }

    */

}
