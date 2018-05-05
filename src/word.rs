#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Word(u64);

impl From<u64> for Word {
    fn from(i: u64) -> Self {
        Word(i)
    }
}

impl From<Word> for u64 {
    fn from(i: Word) -> Self {
        i.0
    }
}

impl Word {
    pub const ZEROS: Self = Word(0);

    #[inline]
    fn len(self) -> usize {
        64
    }

    #[inline]
    fn index_check(self, idx: usize) -> Option<()> {
        if idx >= self.len() { None } else { Some(()) }
    }

    #[inline]
    pub fn get(self, idx: usize) -> Option<bool> {
        self.index_check(idx)?;
        Some((u64::from(self) & (1 << (63 - idx))) > 0)
    }

    pub fn set_copy(self, idx: usize, to: bool) -> Option<Self> {
        self.index_check(idx)?;
        let mask = 1 << (63 - idx);
        let int = u64::from(self);
        let res = if to { int | mask } else { int & (!mask) };
        Some(res.into())
    }

    #[inline]
    pub fn set(&mut self, idx: usize, to: bool) -> Result<(), &'static str> {
        match self.set_copy(idx, to) {
            None => Err("index out of bounds"),
            Some(rep) => Ok(*self = rep),
        }
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

impl Word {
    #[inline]
    pub fn count_ones(self) -> u32 {
        u64::from(self).count_ones()
    }

    #[inline]
    pub fn count_zeros(self) -> u32 {
        u64::from(self).count_zeros()
    }

    #[inline]
    pub fn rank(self, idx: usize) -> Option<u32> {
        if idx == 0 {
            return Some(0);
        };
        self.index_check(idx)?;
        let to_count = u64::from(self) >> (64 - idx);
        Some(to_count.count_ones())
    }

    #[inline]
    pub fn complement(self) -> Self {
        Self::from(!u64::from(self))
    }

    #[inline]
    pub fn rank_zeros(self, idx: usize) -> Option<u32> {
        self.complement().rank(idx)
    }

    pub fn select(self, nth: u32) -> Option<u32> {
        let rank_32 = self.rank(32)?;
        let rank_16 = self.rank(16)?;
        let rank_48 = self.rank(48)?;
        let int: u64 = self.into();

        let res = if rank_32 > nth {
            if rank_16 > nth {
                select_u16(((int >> 48) & 0xffff) as u16, nth)
            } else {
                select_u16(((int >> 32) & 0xffff) as u16, nth - rank_16) + 16
            }
        } else {
            if rank_48 > nth {
                select_u16(((int >> 16) & 0xffff) as u16, nth - rank_32) + 32
            } else {
                if nth >= self.count_ones() {
                    return None;
                }

                select_u16((int & 0xffff) as u16, nth - rank_48) + 48
            }
        };
        Some(res)
    }

    #[inline]
    pub fn select_zeros(self, nth: u32) -> Option<u32> {
        self.complement().select(nth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck;
    use quickcheck::Arbitrary;

    impl Arbitrary for Word {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            u64::arbitrary(g).into()
        }

        fn shrink(&self) -> Box<Iterator<Item = Self>> {
            let base = u64::from(*self).shrink();
            Box::new(base.map(|x| x.into()))
        }
    }

    #[test]
    fn test_get() {
        let get_bits = |x: u64| {
            let x = Word::from(x);
            let mut res = Vec::with_capacity(x.len());
            for i in 0..x.len() {
                res.push(x.get(i).unwrap());
            }
            res.into_iter().enumerate()
        };
        for (i, x) in get_bits(0xffffffffffffffff) {
            assert!(x, "Bit {} was wrong", i);
        }
        for (i, x) in get_bits(0x0000000000000000) {
            assert!(!x, "Bit {} was wrong", i);
        }
        for (i, x) in get_bits(0xffffffff00000000) {
            assert_eq!(x, i < 32, "Bit {} was wrong", i);
        }
        for (i, x) in get_bits(0x00000000ffffffff) {
            assert_eq!(x, i >= 32, "Bit {} was wrong", i);
        }
        for (i, x) in get_bits(0xaaaaaaaaaaaaaaaa) {
            assert_eq!(x, (i & 1) == 0, "Bit {} was wrong", i);
        }
        for (i, x) in get_bits(0x5555555555555555) {
            assert_eq!(x, (i & 1) > 0, "Bit {} was wrong", i);
        }

        for shift in 0..Word::from(0).len() {
            for (i, x) in get_bits(0x8000000000000000 >> shift) {
                assert_eq!(x, i == shift, "Bit {} was wrong", i);
            }
        }
    }

    quickcheck! {
        fn bounds_check_get(x: Word) -> bool {
            x.get(x.len()).is_none()
        }

        fn test_set(x: Word) -> bool {
            (0..x.len()).all(|i| {
                let set_true = {
                    let mut x = x.clone();
                    x.set(i, true).unwrap();
                    x
                };
                let set_false = {
                    let mut x = x.clone();
                    x.set(i, false).unwrap();
                    x
                };

                (0..x.len()).all(|j| {
                    if i == j {
                        set_true.get(j).unwrap() == true
                        && set_false.get(j).unwrap() == false
                    } else {
                        set_true.get(j) == x.get(j)
                        && set_false.get(j) == x.get(j)
                    }
                })
            })
        }

        fn bounds_check_set(x: Word, to: bool) -> bool {
            let mut x = x;
            let len = x.len();
            x.set(len, to).is_err()
        }

        fn test_count_ones(x: Word) -> bool {
            let expected = (0..64).filter(|&i| x.get(i).unwrap()).count();
            x.count_ones() as usize == expected
        }

        fn test_count_zeros(x: Word) -> bool {
            let expected = (0..64).filter(|&i| !x.get(i).unwrap()).count();
            x.count_zeros() as usize == expected
        }

        fn test_rank(x: Word) -> bool {
            for i in 0..x.len() {
                let expected = (0..i).filter(|&i| x.get(i).unwrap()).count();
                assert_eq!(x.rank(i).unwrap() as usize, expected, "Rank didn't match at {}", i);
            }
            true
        }

        fn bounds_check_rank(x: Word) -> bool {
            x.rank(x.len()).is_none()
        }

        fn test_rank_zeros(x: Word) -> bool {
            for i in 0..x.len() {
                let expected = (0..i).filter(|&i| !x.get(i).unwrap()).count();
                assert_eq!(x.rank_zeros(i).unwrap() as usize, expected, "Rank didn't match at {}", i);
            }
            true
        }

        fn bounds_check_rank_zeros(x: Word) -> bool {
            x.rank_zeros(x.len()).is_none()
        }

        fn test_complement(x: Word) -> bool {
            let y = x.complement();
            (0..x.len()).all(|i| x.get(i) != y.get(i))
        }

        fn test_select_u16(x: u16) -> bool {
            let total_count = x.count_ones() as usize;
            let x_bits = Word::from((x as u64) << 48);
            for i in 0..total_count {
                let r = select_u16(x, i as u32) as usize;
                let prev_rank = (0..r).filter(|&j| x_bits.get(j).unwrap()).count();
                assert_eq!(prev_rank, i);
                assert!(x_bits.get(r).unwrap());
            }
            assert_eq!(select_u16(x, total_count as u32), 16);
            assert_eq!(select_u16(x, 16), 16);
            true
        }

        fn test_select(x: Word) -> bool {
            let total_count = x.count_ones() as usize;
            for i in 0..total_count {
                let r = x.select(i as u32).unwrap() as usize;
                assert_eq!(x.rank(r).unwrap() as usize, i);
                assert!(x.get(r).unwrap());
            }
            true
        }

        fn bounds_check_select(x: Word) -> bool {
            x.select(x.count_ones()).is_none()
        }

        fn test_select_zeros(x: Word) -> bool {
            let total_count = x.count_zeros() as usize;
            for i in 0..total_count {
                let r = x.select_zeros(i as u32).unwrap() as usize;
                assert_eq!(x.rank_zeros(r).unwrap() as usize, i);
                assert!(!x.get(r).unwrap());
            }
            true
        }

        fn bounds_check_select_zeros(x: Word) -> bool {
            x.select_zeros(x.count_zeros()).is_none()
        }
    }
}
