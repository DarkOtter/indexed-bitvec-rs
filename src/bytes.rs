use super::{div_64, mod_64, mult_64, ceil_div, MAX_BITS, MAX_BITS_IN_BYTES};
use std::mem::size_of;
use std::cmp::min;
use word::{select_ones_u16, Word};
use byteorder::{BigEndian, ByteOrder};
use ones_or_zeros::OnesOrZeros;

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

fn bytes_as_u64s(data: &[u8]) -> (&[u8], &[u64], &[u8]) {
    // For actually casting we must match alignment
    let alignment_offset = {
        use std::mem::align_of;
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

    (pre_partial, data, post_partial)
}

pub fn count_ones(data: &[u8]) -> Option<u64> {
    if data.len() > MAX_BITS_IN_BYTES {
        // Avoid overflow
        return None;
    }

    let (pre_partial, data, post_partial) = bytes_as_u64s(data);
    let pre_partial = pre_partial
        .iter()
        .map(|&x| x.count_ones() as u64)
        .sum::<u64>();
    let post_partial = post_partial
        .iter()
        .map(|&x| x.count_ones() as u64)
        .sum::<u64>();
    let data = data.iter().map(|&x| x.count_ones() as u64).sum::<u64>();
    Some(pre_partial + data + post_partial)
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

fn len_in_bits<T: Sized>(dat: &[T]) -> u64 {
    use std::mem::size_of;
    dat.len() as u64 * size_of::<T>() as u64 * 8
}

pub fn select<W: OnesOrZeros>(data: &[u8], idx: u64) -> Option<u64> {
    let (pre_partial, data, post_partial) = bytes_as_u64s(data);

    let mut running_count = 0u64;
    let mut running_index = 0u64;

    for &byte in pre_partial.iter() {
        let count = W::convert_count(byte.count_ones() as u64, 8);
        if running_count + count > idx {
            let selected = select_ones_u16(byte as u16, (idx - running_count) as u32);
            let answer = selected as u64 - 8 + running_index;
            return Some(answer);
        }
        running_count += count;
        running_index += 8;
    }

    for &word in data.iter() {
        let count = W::convert_count(word.count_ones() as u64, 64);
        if running_count + count > idx {
            let answer = Word::from(u64::from_be(word))
                .select::<W>((idx - running_count) as u32)
                .map(|sub_res| running_count + sub_res as u64);
            return answer;
        }
        running_count += count;
        running_index += 64;
    }

    for &byte in post_partial.iter() {
        let count = W::convert_count(byte.count_ones() as u64, 8);
        if running_count + count > idx {
            let selected = select_ones_u16(byte as u16, (idx - running_count) as u32);
            let answer = selected as u64 - 8 + running_index;
            return Some(answer);
        }
        running_count += count;
        running_index += 8;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

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
