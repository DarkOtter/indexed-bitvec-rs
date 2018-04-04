use bitvec64::Bits64;

fn select_u8(mut from: u8, mut nth: u32) -> u32 {
    let mut res = 0;
    loop {
        if (from & 1) > 0 {
            if nth == 0 {
                return res;
            } else {
                nth -= 1;
            }
        }
        from >>= 1;
        res += 1;
    }
}

impl Bits64 {
    pub fn count_ones(self) -> u32 {
        u64::from(self).count_ones()
    }

    pub fn count_zeros(self) -> u32 {
        u64::from(self).count_zeros()
    }

    pub fn rank(self, idx: usize) -> u32 {
        if idx >= 64 {
            return self.count_ones();
        };
        let to_count = u64::from(self) << (64 - idx);
        to_count.count_ones()
    }

    pub fn complement(self) -> Self {
        Self::from(!u64::from(self))
    }

    pub fn rank_zeros(self, idx: usize) -> u32 {
        self.complement().rank(idx)
    }

    pub fn select(self, nth: u32) -> u32 {
        let mut nth = nth;
        let mut from = u64::from(self);
        let mut res = 0;

        let part_pop = (from & 0xffffffff).count_ones();
        if part_pop <= nth {
            nth -= part_pop;
            from >>= 32;
            res += 32;
        }

        let part_pop = (from & 0xffff).count_ones();
        if part_pop <= nth {
            nth -= part_pop;
            from >>= 16;
            res += 16;
        }

        let part_pop = (from & 0xff).count_ones();
        if part_pop <= nth {
            nth -= part_pop;
            from >>= 8;
            res += 8;
        }

        let from = (from & 0xff) as u8;
        res + select_u8(from, nth)
    }

    pub fn select_zero(self, nth: u32) -> u32 {
        self.complement().select(nth)
    }
}
