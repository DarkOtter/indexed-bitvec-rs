#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
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
    fn len(self) -> usize {
        64
    }

    #[inline]
    fn index_check(self, idx: usize) {
        if idx >= self.len() {
            panic!("Index out of range");
        }
    }

    #[inline]
    pub fn get(self, idx: usize) -> bool {
        self.index_check(idx);
        (u64::from(self) & (1 << (63 - idx))) > 0
    }

    pub fn set_copy(self, idx: usize, to: bool) -> Self {
        self.index_check(idx);
        let mask = 1 << (63 - idx);
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
        self.index_check(idx);
        if idx == 0 {
            return 0;
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
                if nth >= self.count_ones() {
                    panic!("Index out of range");
                }

                select_u16((int & 0xffff) as u16, nth - rank_48) + 48
            }
        }
    }

    #[inline]
    pub fn select_zeros(self, nth: u32) -> u32 {
        self.complement().select(nth)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck;
    use quickcheck::Arbitrary;

    impl Arbitrary for Bits64 {
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
            let x = Bits64::from(x);
            let mut res = Vec::with_capacity(x.len());
            for i in 0..x.len() {
                res.push(x.get(i));
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

        for shift in 0..Bits64::from(0).len() {
            for (i, x) in get_bits(0x8000000000000000 >> shift) {
                assert_eq!(x, i == shift, "Bit {} was wrong", i);
            }
        }
    }

    quickcheck! {
        fn bounds_check_get(x: Bits64) -> bool {
            use std::panic::catch_unwind;
            catch_unwind(|| x.get(x.len())).is_err()
        }

        fn test_set(x: Bits64) -> bool {
            (0..x.len()).all(|i| {
                let set_true = {
                    let mut x = x.clone();
                    x.set(i, true);
                    x
                };
                let set_false = {
                    let mut x = x.clone();
                    x.set(i, false);
                    x
                };

                (0..x.len()).all(|j| {
                    if i == j {
                        set_true.get(j) == true
                        && set_false.get(j) == false
                    } else {
                        set_true.get(j) == x.get(j)
                        && set_false.get(j) == x.get(j)
                    }
                })
            })
        }

        fn bounds_check_set(x: Bits64, to: bool) -> bool {
            use std::panic::catch_unwind;
            catch_unwind(|| {
                let mut x = x;
                let len = x.len();
                x.set(len, to)
            }).is_err()
        }

        fn test_count_ones(x: Bits64) -> bool {
            let expected = (0..64).filter(|&i| x.get(i)).count();
            x.count_ones() as usize == expected
        }

        fn test_count_zeros(x: Bits64) -> bool {
            let expected = (0..64).filter(|&i| !x.get(i)).count();
            x.count_zeros() as usize == expected
        }

        fn test_rank(x: Bits64) -> bool {
            for i in 0..x.len() {
                let expected = (0..i).filter(|&i| x.get(i)).count();
                assert_eq!(x.rank(i) as usize, expected, "Rank didn't match at {}", i);
            }
            true
        }

        fn bounds_check_rank(x: Bits64) -> bool {
            use std::panic::catch_unwind;
            catch_unwind(|| x.rank(x.len())).is_err()
        }

        fn test_rank_zeros(x: Bits64) -> bool {
            for i in 0..x.len() {
                let expected = (0..i).filter(|&i| !x.get(i)).count();
                assert_eq!(x.rank_zeros(i) as usize, expected, "Rank didn't match at {}", i);
            }
            true
        }

        fn bounds_check_rank_zeros(x: Bits64) -> bool {
            use std::panic::catch_unwind;
            catch_unwind(|| x.rank_zeros(x.len())).is_err()
        }

        fn test_complement(x: Bits64) -> bool {
            let y = x.complement();
            (0..x.len()).all(|i| x.get(i) != y.get(i))
        }

        fn test_select_u16(x: u16) -> bool {
            let total_count = x.count_ones() as usize;
            let x_bits = Bits64::from((x as u64) << 48);
            for i in 0..total_count {
                let r = select_u16(x, i as u32) as usize;
                let prev_rank = (0..r).filter(|&j| x_bits.get(j)).count();
                assert_eq!(prev_rank, i);
                assert!(x_bits.get(r));
            }
            assert_eq!(select_u16(x, total_count as u32), 16);
            assert_eq!(select_u16(x, 16), 16);
            true
        }

        fn test_select(x: Bits64) -> bool {
            let total_count = x.count_ones() as usize;
            for i in 0..total_count {
                let r = x.select(i as u32) as usize;
                assert_eq!(x.rank(r) as usize, i);
                assert!(x.get(r));
            }
            true
        }

        fn bounds_check_select(x: Bits64) -> bool {
            use std::panic::catch_unwind;
            catch_unwind(|| x.select(x.count_ones())).is_err()
        }

        fn test_select_zeros(x: Bits64) -> bool {
            let total_count = x.count_zeros() as usize;
            for i in 0..total_count {
                let r = x.select_zeros(i as u32) as usize;
                assert_eq!(x.rank_zeros(r) as usize, i);
                assert!(!x.get(r));
            }
            true
        }

        fn bounds_check_select_zeros(x: Bits64) -> bool {
            use std::panic::catch_unwind;
            catch_unwind(|| x.select_zeros(x.count_zeros())).is_err()
        }
    }

}
