use super::{div_64, mod_64, mult_64};
use bits64::Bits64;
use byteorder::{BigEndian, ByteOrder};

fn read_partial_word(data: &[u8]) -> u64 {
    if data.len() >= 8 {
        panic!("Expected partial word, got at least a full word");
    }
    let mut res = [0u8; 8];
    res[0..data.len()].copy_from_slice(data);
    BigEndian::read_u64(&res)
}

pub fn read_word(data: &[u8], idx: usize) -> u64 {
    let idx_bytes = idx << 3;
    let end_idx_bytes = idx_bytes + 8;
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
    debug_assert!((data.len() - idx_bytes) < 8);
    read_partial_word(&data[idx_bytes..])
}

pub fn len_words(data: &[u8]) -> usize {
    let byte_len = data.len();
    let full_words = byte_len >> 3;
    let partial_word = if (byte_len & 7) > 0 { 1 } else { 0 };
    full_words + partial_word
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
