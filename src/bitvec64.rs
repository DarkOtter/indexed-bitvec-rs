#[inline(always)]
fn mult_64(i: usize) -> usize {
    i << 6
}

#[inline(always)]
fn div_64(i: usize) -> usize {
    i >> 6
}

#[inline(always)]
fn mod_64(i: usize) -> usize {
    i & 63
}

fn ceil_div_64(i: usize) -> usize {
    div_64(i) + (if mod_64(i) > 0 { 1 } else { 0 })
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Bits64(u64);

impl Bits64 {
    pub const ZEROS: Self = Bits64(0);

    fn index_check(idx: usize) {
        if idx >= 64 {
            panic!("Index out of range");
        }
    }

    pub fn get(self, idx: usize) -> bool {
        Self::index_check(idx);
        ((self.0 >> idx) & 1) > 0
    }

    pub fn set(&mut self, idx: usize, to: bool) {
        Self::index_check(idx);
        let mask = 1 << idx;
        let res = if to { self.0 | mask } else { self.0 & (!mask) };
        self.0 = res
    }
}

impl From<u64> for Bits64 {
    fn from(i: u64) -> Self {
        Bits64(i)
    }
}

impl From<Bits64> for u64 {
    fn from(i: Bits64) -> Self {
        i.0
    }
}

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

use byteorder::{ReadBytesExt, WriteBytesExt, NetworkEndian};
use std::io;
use std::io::{Read, Write};

impl Bits64 {
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let res = reader.read_u64::<NetworkEndian>()?;
        Ok(Bits64(res))
    }

    pub fn write_to<W: Write>(self, writer: &mut W) -> io::Result<()> {
        writer.write_u64::<NetworkEndian>(self.0)
    }
}

impl BitVec64 {
    pub fn read_from<R: Read>(reader: &mut R) -> io::Result<Self> {
        let used_bits = reader.read_u64::<NetworkEndian>()?;
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

    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u64::<NetworkEndian>(self.used_bits as u64)?;
        for bits in self.data.iter() {
            bits.write_to(writer)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
