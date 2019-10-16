/*
   Copyright 2018 DarkOtter

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
//! Type to represent bits, and basic count/rank/select functions for it.
use indexed_bitvec_core::bits_ref::BitsRef;
use std::ops::Deref;

/// A bitvector stored as a sequence of bytes (most significant bit first).
#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
#[serde(remote = "Self")]
pub struct Bits<T: Deref<Target = [u8]>>((T, u64));

impl<T: serde::Serialize + Deref<Target = [u8]>> serde::Serialize for Bits<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        Bits::serialize(self, serializer)
    }
}

impl<'de, T: serde::Deserialize<'de> + Deref<Target = [u8]>> serde::Deserialize<'de> for Bits<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (bytes, len) = Bits::deserialize(deserializer)?.decompose();
        Bits::from_bytes(bytes, len).ok_or_else(|| serde::de::Error::custom("Invalid bits data"))
    }
}

impl<'a, T: Deref<Target = [u8]>> From<&'a Bits<T>> for BitsRef<'a> {
    fn from(bits: &'a Bits<T>) -> Self {
        let len = (bits.0).1;
        BitsRef::from_bytes((bits.0).0.deref(), len).expect("Bits with invalid len")
    }
}

impl<'a> From<Bits<&'a [u8]>> for BitsRef<'a> {
    fn from(bits: Bits<&'a [u8]>) -> Self {
        let len = (bits.0).1;
        BitsRef::from_bytes((bits.0).0, len).expect("Bits with invalid len")
    }
}

impl<'a> From<BitsRef<'a>> for Bits<&'a [u8]> {
    fn from(bits: BitsRef<'a>) -> Self {
        Bits(bits.into())
    }
}

impl<T: Deref<Target = [u8]>> Bits<T> {
    pub fn from_bytes(bytes: T, len: u64) -> Option<Self> {
        // Use BitsRef to check the size
        match BitsRef::from_bytes(bytes.deref(), len) {
            None => None,
            Some(_) => Some(Bits((bytes, len))),
        }
    }

    /// All of the bytes stored in the byte sequence: not just the ones actually used.
    #[inline]
    pub fn all_bytes(&self) -> &[u8] {
        BitsRef::from(self).all_bytes()
    }

    /// The number of bits in the storage.
    #[inline]
    pub fn len(&self) -> u64 {
        (self.0).1
    }

    /// The used bytes of the byte sequence: bear in mind some of the bits in the
    /// last byte may be unused.
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        BitsRef::from(self).bytes()
    }

    /// Deconstruct the bits storage to get back what it was constructed from.
    #[inline]
    pub fn decompose(self) -> (T, u64) {
        self.0
    }

    /// Get the byte at a specific index.
    ///
    /// Returns `None` for out-of-bounds.
    ///
    /// ```
    /// use indexed_bitvec::Bits;
    /// let bits = Bits::from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.get(0), Some(true));
    /// assert_eq!(bits.get(7), Some(false));
    /// assert_eq!(bits.get(14), Some(true));
    /// assert_eq!(bits.get(15), None);
    /// ```
    #[inline]
    pub fn get(&self, idx_bits: u64) -> Option<bool> {
        BitsRef::from(self).get(idx_bits)
    }

    /// Count the set bits (*O(n)*).
    ///
    /// ```
    /// use indexed_bitvec::Bits;
    /// let bits = Bits::from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.count_ones(), 14);
    /// assert_eq!(bits.count_zeros(), 1);
    /// assert_eq!(bits.count_ones() + bits.count_zeros(), bits.len());
    /// ```
    pub fn count_ones(&self) -> u64 {
        BitsRef::from(self).count_ones()
    }

    /// Count the unset bits (*O(n)*).
    ///
    /// ```
    /// use indexed_bitvec::Bits;
    /// let bits = Bits::from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.count_ones(), 14);
    /// assert_eq!(bits.count_zeros(), 1);
    /// assert_eq!(bits.count_ones() + bits.count_zeros(), bits.len());
    /// ```
    #[inline]
    pub fn count_zeros(&self) -> u64 {
        BitsRef::from(self).count_zeros()
    }

    /// Count the set bits before a position in the bits (*O(n)*).
    ///
    /// Returns `None` it the index is out of bounds.
    ///
    /// ```
    /// use indexed_bitvec::Bits;
    /// let bits = Bits::from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert!((0..bits.len()).all(|idx|
    ///     bits.rank_ones(idx).unwrap()
    ///     + bits.rank_zeros(idx).unwrap()
    ///     == (idx as u64)));
    /// assert_eq!(bits.rank_ones(7), Some(7));
    /// assert_eq!(bits.rank_zeros(7), Some(0));
    /// assert_eq!(bits.rank_ones(8), Some(7));
    /// assert_eq!(bits.rank_zeros(8), Some(1));
    /// assert_eq!(bits.rank_ones(9), Some(8));
    /// assert_eq!(bits.rank_zeros(9), Some(1));
    /// assert_eq!(bits.rank_ones(15), None);
    /// ```
    pub fn rank_ones(&self, idx: u64) -> Option<u64> {
        BitsRef::from(self).rank_ones(idx)
    }

    /// Count the unset bits before a position in the bits (*O(n)*).
    ///
    /// Returns `None` it the index is out of bounds.
    ///
    /// ```
    /// use indexed_bitvec::Bits;
    /// let bits = Bits::from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert!((0..bits.len()).all(|idx|
    ///     bits.rank_ones(idx).unwrap()
    ///     + bits.rank_zeros(idx).unwrap()
    ///     == (idx as u64)));
    /// assert_eq!(bits.rank_ones(7), Some(7));
    /// assert_eq!(bits.rank_zeros(7), Some(0));
    /// assert_eq!(bits.rank_ones(8), Some(7));
    /// assert_eq!(bits.rank_zeros(8), Some(1));
    /// assert_eq!(bits.rank_ones(9), Some(8));
    /// assert_eq!(bits.rank_zeros(9), Some(1));
    /// assert_eq!(bits.rank_ones(15), None);
    /// ```
    #[inline]
    pub fn rank_zeros(&self, idx: u64) -> Option<u64> {
        BitsRef::from(self).rank_zeros(idx)
    }

    /// Find the position of a set bit by its rank (*O(n)*).
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank_ones(result) == Some(target_rank)`
    /// and `get(result) == Some(true)`.
    ///
    /// ```
    /// use indexed_bitvec::Bits;
    /// let bits = Bits::from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.select_ones(6), Some(6));
    /// assert_eq!(bits.select_ones(7), Some(8));
    /// assert_eq!(bits.select_zeros(0), Some(7));
    /// assert_eq!(bits.select_zeros(1), None);
    /// ```
    pub fn select_ones(&self, target_rank: u64) -> Option<u64> {
        BitsRef::from(self).select_ones(target_rank)
    }

    /// Find the position of an unset bit by its rank (*O(n)*).
    ///
    /// Returns `None` if no suitable bit is found. It is
    /// always the case otherwise that `rank_zeros(result) == Some(target_rank)`
    /// and `get(result) == Some(false)`.
    ///
    /// ```
    /// use indexed_bitvec::Bits;
    /// let bits = Bits::from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.select_ones(6), Some(6));
    /// assert_eq!(bits.select_ones(7), Some(8));
    /// assert_eq!(bits.select_zeros(0), Some(7));
    /// assert_eq!(bits.select_zeros(1), None);
    /// ```
    pub fn select_zeros(&self, target_rank: u64) -> Option<u64> {
        BitsRef::from(self).select_zeros(target_rank)
    }

    /// Create a reference to these same bits.
    pub fn clone_ref(&self) -> Bits<&[u8]> {
        Bits::from_bytes(self.all_bytes(), self.len()).expect("Bits with invalid len")
    }
}

impl<T: core::ops::DerefMut<Target = [u8]>> Bits<T> {
    #[inline]
    pub fn all_bytes_mut(&mut self) -> &mut [u8] {
        (self.0).0.deref_mut()
    }

    /// Set the byte at a specific index.
    ///
    /// Returns an error if the index is out of bounds.
    ///
    /// ```
    /// use indexed_bitvec::Bits;
    /// let mut bits = Bits::from_bytes(vec![0xFE, 0xFE], 15).unwrap();
    /// assert_eq!(bits.get(0), Some(true));
    /// assert_eq!(bits.get(7), Some(false));
    /// assert_eq!(bits.get(14), Some(true));
    /// assert_eq!(bits.get(15), None);
    /// assert!(bits.set(0, false).is_ok());
    /// assert_eq!(bits.get(0), Some(false));
    /// assert!(bits.set(0, true).is_ok());
    /// assert_eq!(bits.get(0), Some(true));
    /// assert!(bits.set(7, false).is_ok());
    /// assert_eq!(bits.get(7), Some(false));
    /// assert!(bits.set(14, false).is_ok());
    /// assert_eq!(bits.get(14), Some(false));
    /// assert!(bits.set(15, false).is_err());
    /// ```
    pub fn set(&mut self, idx_bits: u64, to: bool) -> Result<(), &'static str> {
        let len = self.len();
        if idx_bits >= len {
            Err("Index out-of-bounds")
        } else {
            let data = self.all_bytes_mut();
            let byte_idx = (idx_bits / 8) as usize;
            let idx_in_byte = (idx_bits % 8) as usize;

            let mask = 0x80 >> idx_in_byte;

            if to {
                data[byte_idx] |= mask
            } else {
                data[byte_idx] &= !mask
            }

            Ok(())
        }
    }
}

impl Bits<Vec<u8>> {
    /// Add a specific bit to the end of the vector.
    ///
    /// This will enlarge the used section of bits (corresponding
    /// to `Bits::len`) and overwrite any content already in the
    /// newly used storage.
    ///
    /// ```
    /// use indexed_bitvec::Bits;
    /// let mut bits = Bits::from_bytes(vec![0x80], 0).unwrap();
    /// assert_eq!(0x80, bits.all_bytes()[0]);
    /// assert_eq!(None, bits.get(0));
    /// bits.push(false);
    /// assert_eq!(0x00, bits.all_bytes()[0]);
    /// assert_eq!(Some(false), bits.get(0));
    /// bits.push(true);
    /// assert_eq!(0x40, bits.all_bytes()[0]);
    /// assert_eq!(Some(true), bits.get(1));
    /// for _ in 0..6 { bits.push(false) };
    /// assert_eq!(0x40, bits.all_bytes()[0]);
    /// assert_eq!(1, bits.all_bytes().len());
    /// bits.push(true);
    /// assert_eq!(2, bits.all_bytes().len());
    /// assert_eq!(0x80, bits.all_bytes()[1]);
    /// assert_eq!(Some(true), bits.get(8));
    /// ```
    pub fn push(&mut self, bit: bool) {
        let original_len = (self.0).1;
        (self.0).1 += 1;

        debug_assert!(original_len / 8 <= (self.0).0.len() as u64);
        if original_len % 8 == 0 && original_len / 8 == (self.0).0.len() as u64 {
            (self.0).0.push(if bit { 0x80 } else { 0x00 })
        } else {
            self.set(original_len, bit).expect("Should not be in range");
        }
    }
}

fn must_have_or_bug<T>(opt: Option<T>) -> T {
    opt.expect("If this is None there is a bug in Bits implementation")
}

use std::cmp::Ordering;

impl<T: Deref<Target = [u8]>> std::cmp::Ord for Bits<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        BitsRef::from(self).cmp(&BitsRef::from(other))
    }
}

impl<T: Deref<Target = [u8]>> std::cmp::PartialOrd for Bits<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Deref<Target = [u8]>> std::cmp::Eq for Bits<T> {}

impl<T: Deref<Target = [u8]>> std::cmp::PartialEq for Bits<T> {
    fn eq(&self, other: &Self) -> bool {
        BitsRef::from(self).eq(&BitsRef::from(other))
    }
}

#[cfg(feature = "implement_heapsize")]
impl<T: Deref<Target = [u8]> + heapsize::HeapSizeOf> heapsize::HeapSizeOf for Bits<T> {
    fn heap_size_of_children(&self) -> usize {
        (self.0).0.heap_size_of_children()
    }
}

/// An iterator through individual bits of a bitvector.
#[derive(Copy, Clone, Debug)]
pub struct BitIterator<T: Deref<Target = [u8]>> {
    search_from: u64,
    search_in: Bits<T>,
}

impl<T: Deref<Target = [u8]>> Iterator for BitIterator<T> {
    type Item = bool;

    fn next(&mut self) -> Option<bool> {
        match self.search_in.get(self.search_from) {
            None => None,
            ret => {
                self.search_from += 1;
                ret
            }
        }
    }
}

impl<T: Deref<Target = [u8]>> BitIterator<T> {
    fn next_index<F>(&mut self, with_remaining: F) -> Option<u64>
    where
        F: Fn(Bits<&[u8]>, u64) -> Option<u64>,
    {
        if self.search_from >= self.search_in.len() {
            return None;
        }

        let byte_index = (self.search_from / 8) as usize;
        let byte_offset = self.search_from % 8;

        let byte_index_bits = (byte_index as u64) * 8;

        let remaining_part = must_have_or_bug(Bits::from_bytes(
            &(self.search_in.all_bytes())[byte_index..],
            self.search_in.len() - byte_index_bits,
        ));

        let next_rank = with_remaining(remaining_part, byte_offset);

        match next_rank {
            None => {
                self.search_from = self.search_in.len();
                None
            }
            Some(next_sub_idx) => {
                debug_assert!(next_sub_idx >= byte_offset);
                let res = byte_index_bits + next_sub_idx;
                self.search_from = res + 1;
                Some(res)
            }
        }
    }
}

/// An iterator through the one (set) bit indexes of a bitvector.
#[derive(Copy, Clone, Debug)]
pub struct OneBitIndexIterator<T: Deref<Target = [u8]>>(BitIterator<T>);

impl<T: Deref<Target = [u8]>> Iterator for OneBitIndexIterator<T> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        self.0.next_index(|remaining_part, byte_offset| {
            debug_assert!(byte_offset < 8);
            let target_rank = must_have_or_bug(remaining_part.rank_ones(byte_offset));
            remaining_part.select_ones(target_rank)
        })
    }
}

/// An iterator through the zero (unset) bit indexes of a bitvector.
#[derive(Copy, Clone, Debug)]
pub struct ZeroBitIndexIterator<T: Deref<Target = [u8]>>(BitIterator<T>);

impl<T: Deref<Target = [u8]>> Iterator for ZeroBitIndexIterator<T> {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        self.0.next_index(|remaining_part, byte_offset| {
            debug_assert!(byte_offset < 8);
            let target_rank = must_have_or_bug(remaining_part.rank_zeros(byte_offset));
            remaining_part.select_zeros(target_rank)
        })
    }
}

impl<T: Deref<Target = [u8]>> IntoIterator for Bits<T> {
    type Item = bool;
    type IntoIter = BitIterator<T>;

    fn into_iter(self) -> BitIterator<T> {
        BitIterator {
            search_from: 0,
            search_in: self,
        }
    }
}

impl<T: Deref<Target = [u8]>> Bits<T> {
    pub fn iter(&self) -> BitIterator<&[u8]> {
        self.clone_ref().into_iter()
    }

    pub fn into_iter_one_bits(self) -> OneBitIndexIterator<T> {
        OneBitIndexIterator(self.into_iter())
    }

    pub fn iter_one_bits(&self) -> OneBitIndexIterator<&[u8]> {
        self.clone_ref().into_iter_one_bits()
    }

    pub fn into_iter_zero_bits(self) -> ZeroBitIndexIterator<T> {
        ZeroBitIndexIterator(self.into_iter())
    }

    pub fn iter_zero_bits(&self) -> ZeroBitIndexIterator<&[u8]> {
        self.clone_ref().into_iter_zero_bits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::collection::vec as gen_vec;
    use proptest::collection::SizeRange;
    use proptest::prelude::*;

    type Bitvec = Bits<Vec<u8>>;

    prop_compose! {
        fn gen_bits_inner(byte_len: SizeRange)
            (data in gen_vec(any::<u8>(), byte_len))
            (len in 0..=((data.len() as u64) * 8),
             data in Just(data))
            -> Bitvec
        {
            Bits::from_bytes(data, len).unwrap()
        }
    }

    pub fn gen_bits(byte_len: impl Into<SizeRange>) -> impl Strategy<Value = Bitvec> {
        gen_bits_inner(byte_len.into())
    }

    #[test]
    fn test_basic_from() {
        let example_data = vec![0xff, 0xf0];
        for i in 0..=16 {
            assert!(Bits::from_bytes(example_data.clone(), i).is_some());
        }
        for i in 17..32 {
            assert!(Bits::from_bytes(example_data.clone(), i).is_none());
        }
    }

    fn from_bytes_or_panic<T: Deref<Target = [u8]>>(bytes: T, len: u64) -> Bits<T> {
        Bits::from_bytes(bytes, len).expect("invalid bytes in test")
    }

    proptest! {
        #[test]
        fn test_all_bytes(bits in gen_bits(0..=1024)) {
            let clone_of_data = bits.clone().decompose().0;
            prop_assert_eq!(&clone_of_data[..], bits.all_bytes());
        }

        #[test]
        fn test_bytes(bits in gen_bits(0..=1024)) {
            let need_bytes = ((bits.len() + 7) / 8) as usize;
            prop_assert_eq!(&bits.all_bytes()[..need_bytes], bits.bytes());
        }
    }

    #[test]
    fn test_basic_get() {
        let example_data = vec![0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];
        let max_len = 8 * 8;
        for len in 0..=max_len {
            let bits = from_bytes_or_panic(example_data.clone(), len);
            for i in 0..len {
                assert_eq!(Some(i / 8 == i % 8), bits.get(i));
            }
            for i in len..=(max_len + 1) {
                assert_eq!(None, bits.get(i));
            }
        }

        let example_data = vec![0xff, 0xc0];
        let bits = from_bytes_or_panic(example_data.clone(), 10);
        for i in 0..10 {
            assert_eq!(bits.get(i), Some(true), "Differed at position {}", i)
        }
        for i in 10..16 {
            assert_eq!(bits.get(i), None, "Differed at position {}", i)
        }
    }

    #[test]
    fn test_count_examples() {
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| from_bytes_or_panic(bytes_a, len);
        assert_eq!(12, make(16).count_ones());
        assert_eq!(4, make(16).count_zeros());
        assert_eq!(12, make(15).count_ones());
        assert_eq!(3, make(15).count_zeros());
        assert_eq!(11, make(14).count_ones());
        assert_eq!(3, make(14).count_zeros());
        assert_eq!(11, make(13).count_ones());
        assert_eq!(2, make(13).count_zeros());
        assert_eq!(10, make(12).count_ones());
        assert_eq!(2, make(12).count_zeros());
        assert_eq!(10, make(11).count_ones());
        assert_eq!(1, make(11).count_zeros());
        assert_eq!(9, make(10).count_ones());
        assert_eq!(1, make(10).count_zeros());
        assert_eq!(9, make(9).count_ones());
        assert_eq!(0, make(9).count_zeros());
        assert_eq!(8, make(8).count_ones());
        assert_eq!(0, make(8).count_zeros());
        assert_eq!(7, make(7).count_ones());
        assert_eq!(0, make(7).count_zeros());
        assert_eq!(0, make(0).count_ones());
        assert_eq!(0, make(0).count_zeros());
    }

    fn test_count_via_get(bits: Bitvec, bit_to_count: bool) -> Result<(), TestCaseError> {
        fn inner<F>(bits: Bitvec, bit_to_count: bool, f: F) -> Result<(), TestCaseError>
        where
            F: Fn(&Bitvec) -> u64,
        {
            let count_via_get = (0..bits.len())
                .filter(|&idx| bits.get(idx).unwrap() == bit_to_count)
                .count() as u64;
            prop_assert_eq!(count_via_get, f(&bits));
            Ok(())
        }

        if bit_to_count {
            inner(bits, true, Bits::count_ones)
        } else {
            inner(bits, false, Bits::count_zeros)
        }
    }

    proptest! {
        #[test]
        fn test_count_ones_via_get(bits in gen_bits(0..=1024)) {
            test_count_via_get(bits, true)?;
        }

        #[test]
        fn test_count_ones_via_iter(bits in gen_bits(0..=1024)) {
            let count_via_iter =
                bits.iter()
                .filter(|&b| b)
                .count() as u64;
            prop_assert_eq!(count_via_iter, bits.count_ones());

        }

        #[test]
        fn test_count_zeros_via_get(bits in gen_bits(0..=1024)) {
            test_count_via_get(bits, false)?;
        }

        #[test]
        fn test_count_zeros_via_count_ones(bits in gen_bits(0..=1024)) {
            prop_assert_eq!(bits.len() - bits.count_ones(), bits.count_zeros());
        }

    }

    #[test]
    fn test_rank_examples() {
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| from_bytes_or_panic(bytes_a, len);
        let bits_a = make(16);
        for i in 0..15 {
            assert_eq!(Some(make(i).count_ones()), bits_a.rank_ones(i));
            assert_eq!(Some(make(i).count_zeros()), bits_a.rank_zeros(i));
        }
        assert_eq!(None, bits_a.rank_ones(16));
        assert_eq!(None, bits_a.rank_zeros(16));
        assert_eq!(None, make(13).rank_ones(13));
        assert_eq!(None, make(13).rank_zeros(13));
        assert_eq!(bits_a.rank_ones(12), make(13).rank_ones(12));
        assert_eq!(bits_a.rank_zeros(12), make(13).rank_zeros(12));
    }

    fn test_rank_via_get(bits: Bitvec, bit_to_rank: bool) -> Result<(), TestCaseError> {
        fn inner<F>(bits: Bitvec, bit_to_rank: bool, f: F) -> Result<(), TestCaseError>
        where
            F: Fn(&Bitvec, u64) -> Option<u64>,
        {
            let mut running_rank = 0;
            for idx in 0..=(bits.len() + 64) {
                if idx >= bits.len() {
                    prop_assert_eq!(None, f(&bits, idx), "should be out of range at {}", idx);
                } else {
                    prop_assert_eq!(
                        Some(running_rank),
                        f(&bits, idx),
                        "disagree at index {}",
                        idx
                    );
                    if bits.get(idx).unwrap() == bit_to_rank {
                        running_rank += 1;
                    }
                }
            }
            Ok(())
        }

        if bit_to_rank {
            inner(bits, true, Bits::rank_ones)
        } else {
            inner(bits, false, Bits::rank_zeros)
        }
    }

    proptest! {
        #[test]
        fn test_rank_ones_via_get(bits in gen_bits(0..=1024)) {
            test_rank_via_get(bits, true)?;
        }

        #[test]
        fn test_rank_ones_via_iter(bits in gen_bits(0..=1024)) {
            let mut idx = 0u64;
            let mut running_rank_ones = 0;
            for b in bits.iter() {
                prop_assert_eq!(Some(running_rank_ones), bits.rank_ones(idx),
                                "disagree at index {}", idx);
                idx += 1;
                if b { running_rank_ones += 1 };
            }
            prop_assert_eq!(None, bits.rank_ones(idx),
                            "should be out of range at {}", idx);
        }

        #[test]
        fn test_rank_zeros_via_get(bits in gen_bits(0..=1024)) {
            test_rank_via_get(bits, false)?;
        }

        #[test]
        fn test_rank_zeros_via_rank_ones(bits in gen_bits(0..=1024)) {
            for idx in 0..=(bits.len() + 64) {
                let via_rank_ones =
                    bits.rank_ones(idx).map(|ones| idx - ones);
                prop_assert_eq!(via_rank_ones, bits.rank_zeros(idx));
            }
        }
    }

    #[test]
    fn test_select_examples() {
        let pattern_a = [0xff, 0xaau8];
        let bytes_a = &pattern_a[..];
        let make = |len: u64| from_bytes_or_panic(bytes_a, len);
        assert_eq!(Some(14), make(16).select_ones(11));
        assert_eq!(None, make(14).select_ones(11));
    }

    fn test_select_via_count_rank_get(
        bits: Bitvec,
        bit_to_select: bool,
    ) -> Result<(), TestCaseError> {
        fn inner<C, R, S>(
            bits: Bitvec,
            bit_to_select: bool,
            count: C,
            rank: R,
            select: S,
        ) -> Result<(), TestCaseError>
        where
            C: Fn(&Bitvec) -> u64,
            R: Fn(&Bitvec, u64) -> Option<u64>,
            S: Fn(&Bitvec, u64) -> Option<u64>,
        {
            for bit_idx in 0..count(&bits) {
                let select_idx = select(&bits, bit_idx);
                prop_assert!(
                    select_idx.is_some(),
                    "expected bit_idx to exist: {}",
                    bit_idx
                );
                let select_idx = select_idx.unwrap();
                prop_assert_eq!(
                    Some(bit_idx),
                    rank(&bits, select_idx),
                    "expected rank to be {} at {}",
                    bit_idx,
                    select_idx
                );
                prop_assert_eq!(
                    Some(bit_to_select),
                    bits.get(select_idx),
                    "expected bit to be {} at {}",
                    bit_to_select,
                    select_idx
                );
            }

            prop_assert_eq!(
                None,
                select(&bits, count(&bits)),
                "expected no selected rank for count"
            );

            Ok(())
        }

        if bit_to_select {
            inner(
                bits,
                true,
                Bits::count_ones,
                Bits::rank_ones,
                Bits::select_ones,
            )
        } else {
            inner(
                bits,
                false,
                Bits::count_zeros,
                Bits::rank_zeros,
                Bits::select_zeros,
            )
        }
    }

    proptest! {
        #[test]
        fn test_select_ones_via_count_rank_get(bits in gen_bits(0..=1024)) {
            test_select_via_count_rank_get(bits, true)?;
        }

        #[test]
        fn test_select_zeros_via_count_rank_get(bits in gen_bits(0..=1024)) {
            test_select_via_count_rank_get(bits, false)?;
        }
    }

    #[test]
    fn test_basic_set() {
        let example_data = vec![0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01];
        let max_len = 8 * 8;
        for len in 0..=max_len {
            let bits = from_bytes_or_panic(example_data.clone(), len);
            for set_at in 0..len {
                for set_to in vec![true, false].into_iter() {
                    let mut bits = bits.clone();
                    assert!(bits.set(set_at, set_to).is_ok());
                    let bits = bits;
                    for i in 0..len {
                        if i == set_at {
                            assert_eq!(Some(set_to), bits.get(i));
                        } else {
                            assert_eq!(Some(i / 8 == i % 8), bits.get(i));
                        }
                    }
                    for i in len..=(max_len + 1) {
                        assert_eq!(None, bits.get(i));
                    }
                }
            }
            let mut bits = bits;
            for set_at in len..=(max_len + 1) {
                assert!(bits.set(set_at, true).is_err());
                assert!(bits.set(set_at, false).is_err());
            }
        }
    }

    fn gen_bit_index(bits: &Bitvec) -> BoxedStrategy<Option<u64>> {
        if bits.len() == 0 {
            Just(None).boxed()
        } else {
            (0..bits.len()).prop_map(|x| Some(x)).boxed()
        }
    }

    prop_compose! {
        fn gen_set_task(byte_len: impl Into<SizeRange>)
            (bits in gen_bits(byte_len))
            (idx in gen_bit_index(&bits),
             to in any::<bool>(),
             bits in Just(bits))
             -> (Bitvec, Option<(u64, bool)>) {
                match idx {
                    None => (bits, None),
                    Some(idx) => (bits, Some((idx, to))),
                }
            }
    }

    proptest! {
        #[test]
        fn test_set((bits, task) in gen_set_task(0..=1024)) {
            let (idx, to) = match task {
                None => return Ok(()),
                Some(x) => x,
            };

            let original_bits = bits.clone();
            let mut bits = bits;
            prop_assert_eq!(Ok(()), bits.set(idx, to));

            for check_idx in 0..bits.len() {
                if check_idx == idx {
                    prop_assert_eq!(Some(to), bits.get(check_idx));
                } else {
                    prop_assert_eq!(original_bits.get(check_idx), bits.get(check_idx));
                }
            }
        }
    }

    proptest! {
        #[test]
        fn test_push(bits in gen_bits(0..=1024),
                     add in gen_vec(any::<bool>(), 0..=1024)) {
            let mut as_bool_vec: Vec<_> = bits.iter().collect();
            let mut bits = bits;
            for b in add {
                bits.push(b);
                as_bool_vec.push(b);
            }
            prop_assert_eq!(as_bool_vec, bits.into_iter().collect::<Vec<_>>());
        }
    }

    proptest! {
        #[test]
        fn test_iter_and_into_iter_via_get(bits in gen_bits(0..=1024)) {
            let from_get: Vec<_> = (0..bits.len())
                .map(|idx| bits.get(idx).unwrap())
                .collect();
            let from_iter: Vec<_> = bits.iter().collect();
            let from_into_iter: Vec<_> = bits.into_iter().collect();
            prop_assert_eq!(&from_get, &from_iter);
            prop_assert_eq!(&from_get, &from_into_iter);
        }

        #[test]
        fn test_iter_and_into_iter_one_bits_via_get(bits in gen_bits(0..=1024)) {
            let from_get: Vec<_> = (0..bits.len())
                .filter(|&idx| bits.get(idx).unwrap())
                .collect();
            let from_iter: Vec<_> = bits.iter_one_bits().collect();
            let from_into_iter: Vec<_> = bits.into_iter_one_bits().collect();
            prop_assert_eq!(&from_get, &from_iter);
            prop_assert_eq!(&from_get, &from_into_iter);
        }

        #[test]
        fn test_iter_and_into_iter_zero_bits_via_get(bits in gen_bits(0..=1024)) {
            let from_get: Vec<_> = (0..bits.len())
                .filter(|&idx| !(bits.get(idx).unwrap()))
                .collect();
            let from_iter: Vec<_> = bits.iter_zero_bits().collect();
            let from_into_iter: Vec<_> = bits.into_iter_zero_bits().collect();
            prop_assert_eq!(&from_get, &from_iter);
            prop_assert_eq!(&from_get, &from_into_iter);
        }
    }

    #[test]
    fn test_eq_and_cmp_examples() {
        fn check(expected: Ordering, l: Bitvec, r: Bitvec) {
            let expected_eq = match expected {
                Ordering::Equal => true,
                _ => false,
            };
            assert_eq!(expected_eq, l.eq(&r));
            assert_eq!(expected, l.cmp(&r));
        }

        // Should ignore extra bits
        check(
            Ordering::Equal,
            from_bytes_or_panic(vec![0xff, 0xf0], 12),
            from_bytes_or_panic(vec![0xff, 0xff], 12),
        );

        check(
            Ordering::Equal,
            from_bytes_or_panic(vec![], 0),
            from_bytes_or_panic(vec![], 0),
        );
        check(
            Ordering::Less,
            from_bytes_or_panic(vec![0xff], 0),
            from_bytes_or_panic(vec![0xff], 1),
        );
        check(
            Ordering::Greater,
            from_bytes_or_panic(vec![0xff], 1),
            from_bytes_or_panic(vec![0xff], 0),
        );
        check(
            Ordering::Equal,
            from_bytes_or_panic(vec![0xff], 1),
            from_bytes_or_panic(vec![0xff], 1),
        );
        check(
            Ordering::Less,
            from_bytes_or_panic(vec![0x00], 1),
            from_bytes_or_panic(vec![0xff], 1),
        );
        check(
            Ordering::Greater,
            from_bytes_or_panic(vec![0xff], 1),
            from_bytes_or_panic(vec![0x00], 1),
        );
    }

    fn test_eq_and_cmp_on(l: Bitvec, r: Bitvec) -> Result<(), TestCaseError> {
        let l_as_vec: Vec<_> = l.iter().collect();
        let r_as_vec: Vec<_> = r.iter().collect();
        prop_assert_eq!(l_as_vec.eq(&r_as_vec), l.eq(&r));
        prop_assert_eq!(l_as_vec.cmp(&r_as_vec), l.cmp(&r));
        Ok(())
    }

    proptest! {
        #[test]
        fn test_eq_and_cmp_pair(l in gen_bits(0..=1024), r in gen_bits(0..=1024)) {
            test_eq_and_cmp_on(l, r)?;
        }

        #[test]
        fn test_eq_and_cmp_single(x in gen_bits(0..=1024)) {
            let y = x.clone();
            test_eq_and_cmp_on(x, y)?;
        }
    }

    proptest! {
        #[test]
        fn test_serialise_roundtrip(original in gen_bits(0..=1024)) {
            let serialised = bincode::serialize(&original).unwrap();
            let deserialised = bincode::deserialize(&serialised).unwrap();
            prop_assert_eq!(original, deserialised);
        }
    }
}

#[cfg(test)]
pub use self::tests::gen_bits;
