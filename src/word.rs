use ones_or_zeros::OnesOrZeros;

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

pub(crate) fn select_ones_u16(from: u16, mut nth: u32) -> u32 {
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
    pub fn complement(self) -> Self {
        Self::from(!u64::from(self))
    }

    fn convert_to_ones<W: OnesOrZeros>(self) -> Self {
        if W::is_ones() {
            self
        } else {
            self.complement()
        }
    }

    pub fn count_ones(self) -> u32 {
        u64::from(self).count_ones()
    }

    pub fn count<W: OnesOrZeros>(self) -> u32 {
        self.convert_to_ones::<W>().count_ones()
    }

    #[inline]
    pub fn rank_ones(self, idx: usize) -> Option<u32> {
        if idx == 0 {
            return Some(0);
        };
        self.index_check(idx)?;
        let to_count = u64::from(self) >> (64 - idx);
        Some(to_count.count_ones())
    }

    #[inline]
    pub fn rank<W: OnesOrZeros>(self, idx: usize) -> Option<u32> {
        self.convert_to_ones::<W>().rank_ones(idx)
    }

    pub fn select_ones(self, nth: u32) -> Option<u32> {
        let rank_32 = self.rank_ones(32)?;
        let rank_16 = self.rank_ones(16)?;
        let rank_48 = self.rank_ones(48)?;
        let int: u64 = self.into();

        let res = if rank_32 > nth {
            if rank_16 > nth {
                select_ones_u16(((int >> 48) & 0xffff) as u16, nth)
            } else {
                select_ones_u16(((int >> 32) & 0xffff) as u16, nth - rank_16) + 16
            }
        } else {
            if rank_48 > nth {
                select_ones_u16(((int >> 16) & 0xffff) as u16, nth - rank_32) + 32
            } else {
                if nth >= self.count_ones() {
                    return None;
                }

                select_ones_u16((int & 0xffff) as u16, nth - rank_48) + 48
            }
        };
        Some(res)
    }

    pub fn select<W: OnesOrZeros>(self, nth: u32) -> Option<u32> {
        self.convert_to_ones::<W>().select_ones(nth)
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

    use ones_or_zeros::{OneBits, ZeroBits};

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

        fn test_complement(x: Word) -> bool {
            let y = x.complement();
            (0..x.len()).all(|i| x.get(i) != y.get(i))
        }
    }

    fn do_test_count<W: OnesOrZeros>(x: Word) {
        let expected = (0..64)
            .filter(|&i| x.get(i).unwrap() == W::is_ones())
            .count();
        assert_eq!(x.count::<W>() as usize, expected);
    }

    fn do_test_rank<W: OnesOrZeros>(x: Word) {
        for i in 0..x.len() {
            let expected = (0..i)
                .filter(|&i| x.get(i).unwrap() == W::is_ones())
                .count();
            assert_eq!(
                x.rank::<W>(i).unwrap() as usize,
                expected,
                "Rank didn't match at {}",
                i
            );
        }
        assert!(x.rank::<W>(x.len()).is_none());
    }

    fn do_test_select<W: OnesOrZeros>(x: Word) {
        let total_count = x.count::<W>() as usize;
        for i in 0..total_count {
            let r = x.select::<W>(i as u32).unwrap() as usize;
            assert_eq!(Some(i as u32), x.rank::<W>(r));
            assert_eq!(Some(W::is_ones()), x.get(r));
        }
        assert_eq!(None, x.select::<W>(total_count as u32));
    }

    quickcheck! {
        fn test_count(x: Word) -> () {
            do_test_count::<OneBits>(x);
            do_test_count::<ZeroBits>(x);
        }

        fn test_rank(x: Word) -> () {
            do_test_rank::<OneBits>(x);
            do_test_rank::<OneBits>(x);
        }

        fn test_select_u16(x: u16) -> () {
            let total_count = x.count_ones() as usize;
            let x_bits = Word::from((x as u64) << 48);
            for i in 0..total_count {
                let r = select_ones_u16(x, i as u32) as usize;
                let prev_rank = (0..r).filter(|&j| x_bits.get(j).unwrap()).count();
                assert_eq!(prev_rank, i);
                assert!(x_bits.get(r).unwrap());
            }
            assert_eq!(select_ones_u16(x, total_count as u32), 16);
            assert_eq!(select_ones_u16(x, 16), 16);
        }

        fn test_select(x: Word) -> () {
            do_test_select::<OneBits>(x);
            do_test_select::<ZeroBits>(x);
        }
    }
}
