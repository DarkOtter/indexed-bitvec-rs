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
#[cfg(test)]
extern crate bincode;

#[cfg(test)]
mod tests {
    use indexed_bitvec::{Bits, IndexedBits};

    #[test]
    fn test_succinct_trie_bitvec() {
        // This bitvec was found to break some things that had previously been
        // believed to be invariants of the indexing - specifically the amount
        // of extra samples that might exist in the sampling index.
        let src_data = include_bytes!("../data/strange-cases/succinct-trie.bin");
        let bitvec: Bits<Vec<u8>> = bincode::deserialize(src_data).unwrap();
        let bits = bitvec.clone_ref();
        assert_eq!(1178631, bits.len());
        assert_eq!(589316, bits.count_ones());
        assert_eq!(589315, bits.count_zeros());
        let bits = IndexedBits::build_from_bits(bits);

        let mut running_rank_ones = 0u64;
        let mut running_rank_zeros = 0u64;
        for (idx, bit) in bits.bits().iter().enumerate() {
            let idx = idx as u64;
            assert_eq!(Some(running_rank_ones), bits.rank_ones(idx));
            assert_eq!(Some(running_rank_zeros), bits.rank_zeros(idx));
            if bit {
                assert_eq!(idx, bits.select_ones(running_rank_ones).unwrap());
                running_rank_ones += 1;
            } else {
                assert_eq!(idx, bits.select_zeros(running_rank_zeros).unwrap());
                running_rank_zeros += 1;
            }
        }
    }
}
