use super::{mult_64, div_64, mod_64, ceil_div_64};
use indexable::IndexableData;
use bits64::*;

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

    pub fn data<'a>(&'a self) -> &'a [Bits64] {
        self.data.as_slice()
    }

    pub fn data_mut<'a>(&'a mut self) -> &'a mut [Bits64] {
        self.data.as_mut_slice()
    }
}

impl IndexableData for BitVec64 {
    fn len_bits(&self) -> usize {
        self.len()
    }

    fn get_word(&self, i: usize) -> Bits64 {
        self.data[i]
    }
}

use byteorder::{ReadBytesExt, WriteBytesExt, ByteOrder, BigEndian, NativeEndian};
use std::io;
use std::io::{Read, Write};

impl Bits64 {
    fn from_bytes(bytes: &[u8]) -> Self {
        BigEndian::read_u64(bytes).into()
    }

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

/// Use the bytes of a serialised BitVec64 directly
/// as a read-only vector without copying them.
pub struct SerialisedBitVec64<'a> {
    data: &'a [u8],
    used_bits: usize,
}

impl<'a> SerialisedBitVec64<'a> {
    /// Create a bit vector backed by a bytes slice.
    /// This will return `None` if there are not enough
    /// bytes for the recorded size, or if the recorded
    /// size is too big to represent as a usize
    /// (which can only happen if usize is 32bits).
    /// Otherwise it will also return the number of bytes
    /// used by the bit vector - data after this in the
    /// original slice is not part of the bit vector.
    pub fn of_bytes(data: &'a [u8]) -> Option<(Self, usize)> {
        if data.len() < 8 {
            return None;
        };
        let used_bits = BigEndian::read_u64(data);
        let max_size: usize = !0;
        let max_size: u64 = max_size as u64;
        if used_bits > max_size {
            return None;
        };
        let used_bits: usize = used_bits as usize;
        let units = ceil_div_64(used_bits);
        let data_len = 8 + units * 8;
        if data.len() < data_len {
            return None;
        };
        let res = SerialisedBitVec64 {
            data: &data[8..data_len],
            used_bits,
        };
        Some((res, data_len))
    }

    pub fn len(&self) -> usize {
        self.used_bits
    }

    pub fn get(&self, idx: usize) -> bool {
        if idx >= self.used_bits {
            panic!("Index out of range")
        }
        let word_idx = div_64(idx);
        self.get_word(word_idx).get(mod_64(idx))
    }
}

impl<'a> IndexableData for SerialisedBitVec64<'a> {
    fn len_bits(&self) -> usize {
        self.len()
    }

    fn get_word(&self, i: usize) -> Bits64 {
        let byte_idx = i << 3;
        let end_idx = byte_idx + 8;
        if end_idx > self.data.len() {
            panic!("Index out of range")
        }
        Bits64::from_bytes(&self.data[byte_idx..end_idx])
    }

    fn count_ones_word(&self, i: usize) -> u32 {
        let byte_idx = i << 3;
        let end_idx = byte_idx + 8;
        if end_idx > self.data.len() {
            panic!("Index out of range")
        }
        // We're only counting set bits, so the
        // byte order doesn't matter
        NativeEndian::read_u64(&self.data[byte_idx..end_idx]).count_ones()
    }
}



#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
