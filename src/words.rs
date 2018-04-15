use super::{div_64, mod_64, mult_64};
use std::mem::size_of;
use bits64::Bits64;
use byteorder::{BigEndian, ByteOrder};

fn read_partial_word(data: &[u8]) -> u64 {
    if data.len() >= size_of::<u64>() {
        panic!("Expected partial word, got at least a full word");
    } else if data.len() == 0 {
        return 0;
    }

    let mut res = [0u8; size_of::<u64>()];
    res[0..data.len()].copy_from_slice(data);
    BigEndian::read_u64(&res)
}

pub fn read_word(data: &[u8], idx: usize) -> u64 {
    let idx_bytes = idx * size_of::<u64>();
    let end_idx_bytes = idx_bytes + size_of::<u64>();
    if end_idx_bytes > data.len() {
        read_word_last_word_case(data, idx_bytes)
    } else {
        BigEndian::read_u64(&data[idx_bytes..end_idx_bytes])
    }
}

#[cold]
fn read_word_last_word_case(data: &[u8], idx_bytes: usize) -> u64 {
    if idx_bytes >= data.len() {
        panic!("Index out of bounds");
    }
    debug_assert!((data.len() - idx_bytes) < size_of::<u64>());
    read_partial_word(&data[idx_bytes..])
}

pub fn len_words(data: &[u8]) -> usize {
    let byte_len = data.len();
    let full_words = byte_len / size_of::<u64>();
    let partial_word = if (byte_len % size_of::<u64>()) > 0 {
        1
    } else {
        0
    };
    full_words + partial_word
}

const MAX_SIZE_FOR_U32_COUNT: usize = (<u32>::max_value() / 8) as usize;
const MAX_SIZE_FOR_U64_COUNT: usize = (<u64>::max_value() / 8) as usize;

pub fn count_ones_u32(data: &[u8]) -> u32 {
    if data.len() > MAX_SIZE_FOR_U32_COUNT {
        panic!("Too much data to count into a u32");
    } else if data.len() < size_of::<u64>() {
        return read_partial_word(data).count_ones();
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

    let mut count: u32 = read_partial_word(pre_partial).count_ones() +
        read_partial_word(post_partial).count_ones();

    for word in data {
        count += word.count_ones();
    }

    count
}

pub fn count_ones(data: &[u8]) -> u64 {
    if data.len() > MAX_SIZE_FOR_U64_COUNT {
        panic!("Too much data to count into a u64");
    }

    let mut count: u64 = 0;
    for part in data.chunks(MAX_SIZE_FOR_U32_COUNT) {
        count += count_ones_u32(part) as u64
    }
    count
}

pub trait WordData {
    fn len_words(&self) -> usize;
    fn get_word(&self, idx: usize) -> u64;

    #[inline]
    fn count_ones_word(&self, idx: usize) -> u32 {
        self.get_word(idx).count_ones()
    }
}

impl WordData for [u8] {
    #[inline]
    fn len_words(&self) -> usize {
        len_words(self)
    }

    #[inline]
    fn get_word(&self, idx: usize) -> u64 {
        read_word(self, idx)
    }
}

impl<'a> WordData for &'a [u8] {
    #[inline]
    fn len_words(&self) -> usize {
        (*self).len_words()
    }

    #[inline]
    fn get_word(&self, idx: usize) -> u64 {
        (*self).get_word(idx)
    }

    #[inline]
    fn count_ones_word(&self, idx: usize) -> u32 {
        (*self).count_ones_word(idx)
    }
}

impl WordData for [u64] {
    #[inline]
    fn len_words(&self) -> usize {
        self.len()
    }

    #[inline]
    fn get_word(&self, idx: usize) -> u64 {
        self[idx]
    }
}

impl<'a> WordData for &'a [u64] {
    #[inline]
    fn len_words(&self) -> usize {
        (*self).len_words()
    }

    #[inline]
    fn get_word(&self, idx: usize) -> u64 {
        (*self).get_word(idx)
    }

    #[inline]
    fn count_ones_word(&self, idx: usize) -> u32 {
        (*self).count_ones_word(idx)
    }
}

pub trait WordDataSlice: WordData {
    fn slice_from_word(&self, idx: usize) -> Self;
}

impl<'a> WordDataSlice for &'a [u8] {
    #[inline]
    fn slice_from_word(&self, idx: usize) -> Self {
        &self[idx * 8..]
    }
}

impl<'a> WordDataSlice for &'a [u64] {
    #[inline]
    fn slice_from_word(&self, idx: usize) -> Self {
        &self[idx..]
    }
}


pub trait BitData: WordData {
    fn len_bits(&self) -> usize {
        mult_64(self.len_words())
    }

    fn get_bit(&self, idx: usize) -> bool {
        if idx >= self.len_bits() {
            panic!("Index out of bounds");
        }
        Bits64::from(self.get_word(div_64(idx))).get(mod_64(idx))
    }
}
