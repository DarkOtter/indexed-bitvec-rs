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
//! Operations to create indexes used to perform
//! fast rank & select operations on bitvectors.
//!
//! This crate is still under heavy development,
//! so it will not be very stable in its interface yet.
pub use super::indexed_bitvec_core::parallelism_generic::*;
use super::rayon::{join, scope};

/// Supply to generic functions to execute with some parallelism.
pub enum Parallel {}

impl ExecutionMethod for Parallel {
    #[inline(always)]
    fn do_both<F, G>(f: F, g: G)
    where
        F: FnOnce() + Send,
        G: FnOnce() + Send,
    {
        join(f, g);
    }

    #[inline(always)]
    fn do_many_large_tasks<T, I, F>(iter: I, f: F)
    where
        I: IntoIterator<Item = T> + Send,
        T: Send,
        F: Fn(T) + Send + Sync,
    {
        scope(|s| for x in iter {
            s.spawn(|_| f(x));
        });
    }
}
