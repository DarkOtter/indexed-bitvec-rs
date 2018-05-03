use std::mem;
use std::slice;

#[derive(Debug)]
pub enum ConversionError (
    Misaligned,
    InvalidSize,
);

pub fn bytes_as_u64s(data: &[u8]) -> Result<&[u64], ConversionError> {
    use ConversionError::*;
    unsafe {
        if data.len() % mem::size_of::<u64>() != 0 {
            return Err(InvalidSize);
        }
        let len_u64s = data.len() / mem::size_of::<u64>();

        let ptr = input.as_ptr();
        if (ptr as usize) % mem::align_of::<u64>() != 0 {
            return Err(Misaligned);
        }

        Ok(slice::from_raw_parts(ptr as *const u64, len_u64s))
    }
}
