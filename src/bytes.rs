use super::{div_64, mod_64, mult_64, ceil_div, MAX_BITS, MAX_BITS_IN_BYTES};
use std::mem::size_of;
use std::cmp::min;
use word::Word;
use byteorder::{BigEndian, ByteOrder};
use ones_or_zeros::OnesOrZeros;

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

    let mut count: u64 = {
        let pre_partial =
            read_partial_word(pre_partial).expect("pre_partial should always be a partial word");
        let post_partial =
            read_partial_word(post_partial).expect("post_partial should always be a partial word");
        (pre_partial.count_ones() + post_partial.count_ones()) as u64
    };

    for word in data {
        count += word.count_ones() as u64;
    }

    Some(count)
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

    let mut running_count = {
        let pre_partial_word =
            read_partial_word(pre_partial).expect("pre_partial should always be a partial word");
        let pre_partial_count = pre_partial_word.rank::<W>(pre_partial.len() * 8).expect(
            "index should always be in range",
        ) as u64;
        if pre_partial_count > idx {
            return pre_partial_word.select::<W>(idx as u32).map(Into::into);
        };
        pre_partial_count
    };

    for (i, word) in data.iter().cloned().enumerate() {
        let word_count = W::count(word) as u64;

        if word_count + running_count > idx {
            return Word::from(u64::from_be(word))
                .select::<W>((idx - running_count) as u32)
                .map(|sub_res| {
                    len_in_bits(pre_partial) + len_in_bits(&data[..i]) + sub_res as u64
                });
        };

        running_count += word_count;
    }

    let post_partial_word =
        read_partial_word(post_partial).expect("post_partial should always be a partial word");
    let post_partial_count = post_partial_word.rank::<W>(post_partial.len() * 8).expect(
        "index should always be in range",
    ) as u64;

    if idx >= running_count + post_partial_count {
        return None;
    };

    post_partial_word
        .select::<W>((idx - running_count) as u32)
        .map(|sub_res| {
            len_in_bits(pre_partial) + len_in_bits(data) + sub_res as u64
        })
}