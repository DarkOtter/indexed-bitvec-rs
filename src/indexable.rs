use super::ceil_div_64;
use bits64::Bits64;

pub(crate) trait IndexableData {
    fn len_bits(&self) -> usize;
    fn len_words(&self) -> usize {
        ceil_div_64(self.len_bits())
    }

    fn get_word(&self, i: usize) -> Bits64;
    fn count_ones_word(&self, i: usize) -> u32 {
        self.get_word(i).count_ones()
    }
}
