use super::{mult_64, div_64, mod_64, ceil_div_64};
use words::{WordData, WordDataSlice, BitData};
use bits64::*;

#[derive(Clone, Debug)]
pub struct BitVec64 {
    data: Vec<Bits64>,
    used_bits: usize,
}

impl BitVec64 {
    pub fn new() -> Self {
        BitVec64 {
            data: Vec::new(),
            used_bits: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        BitVec64 {
            data: Vec::with_capacity(ceil_div_64(cap)),
            used_bits: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        mult_64(self.data.capacity())
    }

    pub fn reserve(&mut self, additional: usize) {
        let needed = ceil_div_64(self.used_bits + additional) - self.data.capacity();
        self.data.reserve(needed)
    }

    pub fn get(&self, idx: usize) -> bool {
        if idx >= self.used_bits {
            panic!("Index out of range")
        }

        self.data[div_64(idx)].get(mod_64(idx))
    }

    pub fn set(&mut self, idx: usize, to: bool) {
        if idx >= self.used_bits {
            panic!("Index out of range")
        }

        self.data[div_64(idx)].set(mod_64(idx), to);
    }

    pub fn push(&mut self, val: bool) {
        let idx = self.used_bits;
        if div_64(idx) == 0 {
            self.data.push(Bits64::ZEROS);
        }
        self.used_bits += 1;
        self.data[div_64(idx)].set(mod_64(idx), val);
    }

    pub fn approx_size_bytes(&self) -> usize {
        use std::mem::size_of;
        size_of::<Self>() + self.data.len() * size_of::<Bits64>()
    }

    pub fn len(&self) -> usize {
        self.used_bits
    }
}

impl WordData for BitVec64 {
    fn len_words(&self) -> usize {
        self.data.len()
    }

    fn get_word(&self, idx: usize) -> u64 {
        self.data[idx].into()
    }
}

impl BitData for BitVec64 {
    fn len_bits(&self) -> usize {
        self.len()
    }
}

/// Use the bytes of a serialised BitVec64 directly
/// as a read-only vector without copying them.
#[derive(Clone, Debug)]
pub struct SerialisedBitVec<T: WordData> {
    data: T,
    used_bits: usize,
}

impl<T: WordData> SerialisedBitVec<T> {
    /// Create a bit vector backed by any existing indexable words.
    /// This will return `None` if there are not enough
    /// words for the recorded size, or if the recorded
    /// size is too big to represent as a usize
    /// (which can only happen if usize is 32bits).
    /// Otherwise it will also return the number of words
    /// used by the bit vector - data after this in the
    /// original slice is not part of the bit vector.
    pub fn of_words(data: T) -> Option<(Self, usize)> {
        if data.len_words() < 1 {
            return None;
        };
        let used_bits = data.get_word(0);
        let max_size: usize = !0;
        let max_size: u64 = max_size as u64;
        if used_bits > max_size {
            return None;
        };
        let used_bits: usize = used_bits as usize;
        let words = ceil_div_64(used_bits);
        let used_words = words + 1;
        if data.len_words() < used_words {
            return None;
        };
        let res = SerialisedBitVec { data, used_bits };
        Some((res, used_words))
    }
}

impl<T: WordData> WordData for SerialisedBitVec<T> {
    fn len_words(&self) -> usize {
        ceil_div_64(self.len_bits())
    }

    fn get_word(&self, idx: usize) -> u64 {
        self.data.get_word(idx + 1)
    }

    fn count_ones_word(&self, idx: usize) -> u32 {
        self.data.count_ones_word(idx + 1)
    }
}

impl<T: WordData> BitData for SerialisedBitVec<T> {
    fn len_bits(&self) -> usize {
        self.used_bits
    }

    fn get_bit(&self, idx: usize) -> bool {
        if idx >= self.used_bits {
            panic!("Index out of range")
        }
        Bits64::from(self.data.get_word(div_64(idx))).get(mod_64(idx))
    }
}

impl<T: WordDataSlice> WordDataSlice for SerialisedBitVec<T> {
    fn slice_from_word(&self, idx: usize) -> Self {
        if idx >= self.len_words() {
            panic!("Index out of range");
        }
        SerialisedBitVec {
            data: self.data.slice_from_word(idx),
            used_bits: self.used_bits - mult_64(idx),
        }
    }
}

impl<T: WordData> SerialisedBitVec<T> {
    pub fn len(&self) -> usize {
        <Self as BitData>::len_bits(self)
    }

    pub fn get(&self, idx: usize) -> bool {
        <Self as BitData>::get_bit(self, idx)
    }
}

use byteorder::{ReadBytesExt, WriteBytesExt, BigEndian};
use std::io;
use std::io::{Read, Write};

impl Bits64 {
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let res = reader.read_u64::<BigEndian>()?;
        Ok(res.into())
    }

    pub fn write_to<W: Write>(self, writer: &mut W) -> io::Result<()> {
        writer.write_u64::<BigEndian>(self.into())
    }
}

impl BitVec64 {
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let used_bits = reader.read_u64::<BigEndian>()?;
        let max_size: usize = !0;
        let max_size: u64 = max_size as u64;
        if used_bits > max_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Size is too large to represent on this CPU",
            ));
        }
        let used_bits: usize = used_bits as usize;
        let units = ceil_div_64(used_bits);
        let mut data = Vec::with_capacity(units);
        for _ in 0..units {
            data.push(Bits64::read_from(reader)?);
        }
        Ok(BitVec64 { data, used_bits })
    }

    fn write_data_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        for bits in self.data.iter() {
            bits.write_to(writer)?;
        }
        Ok(())
    }

    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u64::<BigEndian>(self.used_bits as u64)?;
        self.write_data_to(writer)
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut res = Vec::with_capacity(8 * self.data.len());
        self.write_data_to(&mut res).unwrap();
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck;
    use quickcheck::Arbitrary;

    impl Arbitrary for BitVec64 {
        fn arbitrary<G: quickcheck::Gen>(gen: &mut G) -> Self {
            let data = <Vec<Bits64> as Arbitrary>::arbitrary(gen);
            if data.len() == 0 {
                return BitVec64 { data, used_bits: 0 };
            }
            let unused = gen.gen_range(0, 64);
            let used_bits = data.len() * 64 - unused;
            BitVec64 { data, used_bits }
        }
    }

    quickcheck! {
        fn write_then_read(data: BitVec64) -> () {
            let mut buffer = io::Cursor::new(Vec::with_capacity(data.approx_size_bytes()));
            data.write_to(&mut buffer).unwrap();
            let mut buffer = io::Cursor::new(buffer.into_inner());
            let read_back = BitVec64::read_from(&mut buffer).unwrap();
            assert_eq!(data.len(), read_back.len());
            assert_eq!(data.len_bits(), read_back.len_bits());
            assert_eq!(data.len_words(), read_back.len_words());
            for i in 0..data.len_words() {
                assert_eq!(data.get_word(i), read_back.get_word(i));
                assert_eq!(data.count_ones_word(i), read_back.count_ones_word(i));
            }
        }

        fn write_then_use(data: BitVec64) -> () {
            let mut buffer = io::Cursor::new(Vec::new());
            data.write_to(&mut buffer).unwrap();
            let buffer = buffer.into_inner();
            let (reading_back, read_words) =
                SerialisedBitVec::of_words(buffer.as_slice()).unwrap();
            assert_eq!(read_words, buffer.len_words());
            assert_eq!(data.len(), reading_back.len());
            assert_eq!(data.len_bits(), reading_back.len_bits());
            assert_eq!(data.len_words(), reading_back.len_words());
            for i in 0..data.len_words() {
                assert_eq!(data.get_word(i), reading_back.get_word(i));
                assert_eq!(data.count_ones_word(i), reading_back.count_ones_word(i));
            }
        }

        fn write_both_then_use(data_l: BitVec64, data_r: BitVec64) -> () {
            let mut buffer = io::Cursor::new(Vec::new());
            data_l.write_to(&mut buffer).unwrap();
            data_r.write_to(&mut buffer).unwrap();
            let buffer = buffer.into_inner();
            let buffer = buffer.as_slice();
            let (read_l, skip_words) =
                SerialisedBitVec::of_words(buffer).unwrap();
            let (read_r, _) =
                SerialisedBitVec::of_words(buffer.slice_from_word(skip_words)).unwrap();
            assert_eq!(data_l.len(), read_l.len());
            assert_eq!(data_r.len(), read_r.len());
            assert_eq!(data_l.len_bits(), read_l.len_bits());
            assert_eq!(data_r.len_bits(), read_r.len_bits());
            assert_eq!(data_l.len_words(), read_l.len_words());
            assert_eq!(data_r.len_words(), read_r.len_words());

            for i in 0..data_l.len_words() {
                assert_eq!(data_l.get_word(i), read_l.get_word(i));
                assert_eq!(data_l.count_ones_word(i), read_l.count_ones_word(i));
            }

            for i in 0..data_r.len_words() {
                assert_eq!(data_r.get_word(i), read_r.get_word(i));
                assert_eq!(data_r.count_ones_word(i), read_r.count_ones_word(i));
            }
        }
    }
}
