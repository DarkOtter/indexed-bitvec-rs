#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Bits64(u64);

impl Bits64 {
    pub const ZEROS: Self = Bits64(0);

    #[inline]
    fn index_check(idx: usize) {
        if idx >= 64 {
            panic!("Index out of range");
        }
    }

    #[inline]
    pub fn get(self, idx: usize) -> bool {
        Self::index_check(idx);
        (self.0 & (1 << (64 - idx))) > 0
    }

    pub fn set(&mut self, idx: usize, to: bool) {
        Self::index_check(idx);
        let mask = 1 << (64 - idx);
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
