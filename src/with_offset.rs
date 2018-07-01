pub struct WithOffset<T> {
    offset: usize,
    data: T,
}

impl<T> WithOffset<T> {
    pub fn at_origin(data: T) -> Self {
        WithOffset { offset: 0, data }
    }

    pub fn offset_from_origin(&self) -> usize {
        self.offset
    }
}

impl<'a, T> WithOffset<&'a mut [T]> {
    pub fn split_at_mut_by_offset(self, idx: usize) -> (Self, Self) {
        let WithOffset { offset, data } = self;
        if idx < offset {
            panic!("Index out of bounds - before current offset")
        };
        let (data_l, data_r) = data.split_at_mut(idx - offset);
        let data_l_len = data_l.len();
        (
            WithOffset {
                offset,
                data: data_l,
            },
            WithOffset {
                offset: offset + data_l_len,
                data: data_r,
            },
        )
    }

    pub fn take_upto_offset(opt_ref: &mut Option<Self>, idx: usize) -> Option<Self> {
        let whole_thing = match opt_ref.take() {
            None => return None,
            Some(x) => x,
        };
        let (first_part, second_part) = whole_thing.split_at_mut_by_offset(idx);
        *opt_ref = Some(second_part);
        Some(first_part)
    }
}

use std::ops::{Deref, DerefMut};

impl<T> Deref for WithOffset<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for WithOffset<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
