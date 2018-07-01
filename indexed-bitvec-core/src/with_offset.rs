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
//! Use a slice while keeping track of an offset from some logical origin.

/// Store an offset at which the payload data lies.
pub struct WithOffset<T> {
    offset: usize,
    data: T,
}

impl<T> WithOffset<T> {
    /// Start with data at the origin.
    pub fn at_origin(data: T) -> Self {
        WithOffset { offset: 0, data }
    }

    /// Get the current offset from the origin.
    pub fn offset_from_origin(&self) -> usize {
        self.offset
    }
}

impl<'a, T> WithOffset<&'a mut [T]> {
    /// Split the data at an index relative to the origin.
    pub fn split_at_mut_from_origin(self, idx: usize) -> (Self, Self) {
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

    /// Take the first part of the data remaining.
    pub fn take_upto_offset_from_origin(opt_ref: &mut Option<Self>, idx: usize) -> Option<Self> {
        let whole_thing = match opt_ref.take() {
            None => return None,
            Some(x) => x,
        };
        let (first_part, second_part) = whole_thing.split_at_mut_from_origin(idx);
        *opt_ref = Some(second_part);
        Some(first_part)
    }
}

use core::ops::{Deref, DerefMut};

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
