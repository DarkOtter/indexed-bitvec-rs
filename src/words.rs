use super::{div_64, mod_64, mult_64};
use bits64::Bits64;

pub trait WordData {
    fn len_words(&self) -> usize;
    fn get_word(&self, idx: usize) -> u64;
    fn count_ones_word(&self, idx: usize) -> u32 {
        self.get_word(idx).count_ones()
    }
}

fn get_bytes_for_word(data: &[u8], idx: usize) -> [u8; 8] {
    let mut res = [0u8; 8];
    let byte_idx = idx << 3;
    let end_idx = byte_idx + 8;
    if end_idx > data.len() {
        if byte_idx >= data.len() {
            panic!("Index out of bounds");
        } else {
            res[0..data.len() - byte_idx].copy_from_slice(&data[byte_idx..]);
        }
    } else {
        res.copy_from_slice(&data[byte_idx..end_idx])
    }
    res
}

impl WordData for [u8] {
    fn len_words(&self) -> usize {
        let byte_len = self.len();
        let full_words = byte_len >> 3;
        let partial_word = if (byte_len & 7) > 0 { 1 } else { 0 };
        full_words + partial_word
    }

    #[inline]
    fn get_word(&self, idx: usize) -> u64 {
        use byteorder::{BigEndian, ByteOrder};
        BigEndian::read_u64(&get_bytes_for_word(self, idx))
    }

    #[inline]
    fn count_ones_word(&self, idx: usize) -> u32 {
        use byteorder::{NativeEndian, ByteOrder};
        NativeEndian::read_u64(&get_bytes_for_word(self, idx)).count_ones()
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
