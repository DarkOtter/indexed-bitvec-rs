/*
 * A part of indexed-bitvec-rs, a library implementing bitvectors with fast rank operations.
 *     Copyright (C) 2020  DarkOtter
 *
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
