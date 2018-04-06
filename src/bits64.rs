#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Bits64(u64);

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
        (u64::from(self) & (1 << (64 - idx))) > 0
    }

    pub fn set_copy(self, idx: usize, to: bool) -> Self {
        Self::index_check(idx);
        let mask = 1 << (64 - idx);
        let int = u64::from(self);
        let res = if to { int | mask } else { int & (!mask) };
        res.into()
    }

    #[inline]
    pub fn set(&mut self, idx: usize, to: bool) {
        *self = self.set_copy(idx, to)
    }
}

fn select_u16(from: u16, mut nth: u32) -> u32 {
    let mut offset = 0;
    let mut from = from as u32;
    loop {
        let n_leading_zeros = from.leading_zeros() - 16;
        if n_leading_zeros >= 16 {
            return 16;
        } else if nth == 0 {
            return n_leading_zeros + offset;
        };

        let shift = n_leading_zeros + nth;
        from <<= shift;
        offset += shift;
        nth -= (from & 0xffff0000).count_ones();
        from &= 0x0000ffff;
    }
}

impl Bits64 {
    #[inline]
    pub fn count_ones(self) -> u32 {
        u64::from(self).count_ones()
    }

    #[inline]
    pub fn count_zeros(self) -> u32 {
        u64::from(self).count_zeros()
    }

    #[inline]
    pub fn rank(self, idx: usize) -> u32 {
        if idx >= 64 {
            return self.count_ones();
        };
        let to_count = u64::from(self) >> (64 - idx);
        to_count.count_ones()
    }

    #[inline]
    pub fn complement(self) -> Self {
        Self::from(!u64::from(self))
    }

    #[inline]
    pub fn rank_zeros(self, idx: usize) -> u32 {
        self.complement().rank(idx)
    }

    pub fn select(self, nth: u32) -> u32 {
        let rank_32 = self.rank(32);
        let rank_16 = self.rank(16);
        let rank_48 = self.rank(48);
        let int: u64 = self.into();

        if rank_32 > nth {
            if rank_16 > nth {
                select_u16(((int >> 48) & 0xffff) as u16, nth)
            } else {
                select_u16(((int >> 32) & 0xffff) as u16, nth - rank_16) + 16
            }
        } else {
            if rank_48 > nth {
                select_u16(((int >> 16) & 0xffff) as u16, nth - rank_32) + 32
            } else {
                select_u16((int & 0xffff) as u16, nth - rank_48) + 48
            }
        }
    }

    #[inline]
    pub fn select_zero(self, nth: u32) -> u32 {
        self.complement().select(nth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck;
    use quickcheck::{quickcheck, Arbitrary};

    impl Arbitrary for Bits64 {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            u64::arbitrary(g).into()
        }

        fn shrink(&self) -> Box<Iterator<Item = Self>> {
            let base = u64::from(*self).shrink();
            Box::new(base.map(|x| x.into()))
        }
    }
}
