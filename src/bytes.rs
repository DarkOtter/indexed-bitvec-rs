use super::{div_64, mod_64, mult_64, ceil_div};
use std::mem::size_of;
use word::Word;
use byteorder::{BigEndian, ByteOrder};

pub fn read_word(data: &[u8], idx: usize) -> Option<Word> {
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

const MAX_SIZE_FOR_U32_COUNT: usize = (<u32>::max_value() / 8) as usize;
const MAX_SIZE_FOR_U64_COUNT: usize = (<u64>::max_value() / 8) as usize;

pub fn count_ones_u32(data: &[u8]) -> Option<u32> {
    if data.len() > MAX_SIZE_FOR_U32_COUNT {
        return None;
    }

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

    let mut count: u32 = {
        let pre_partial =
            read_partial_word(pre_partial).expect("pre_partial should always be a partial word");
        let post_partial =
            read_partial_word(post_partial).expect("post_partial should always be a partial word");
        pre_partial.count_ones() + post_partial.count_ones()
    };

    for word in data {
        count += word.count_ones();
    }

    Some(count)
}

pub fn count_ones(data: &[u8]) -> Option<u64> {
    if data.len() > MAX_SIZE_FOR_U64_COUNT {
        return None;
    }

    let mut count: u64 = 0;
    for part in data.chunks(MAX_SIZE_FOR_U32_COUNT) {
        count += count_ones_u32(part).expect("sub-part should always be countable in u32") as u64
    }
    Some(count)
}
