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
//! Trait to allow some computations to be generic over parallelism.
//!
//! This is done so that the building can be defined without relying
//! on stdlib, and we can layer parallelism on top in another crate.

pub trait ExecutionMethod {
    /// Run both closures which are independent. Expected to be cheap.
    fn do_both<F, G>(f: F, g: G)
    where
        F: FnOnce() + Send,
        G: FnOnce() + Send;

    /// Run many large computations which are independent.
    fn do_many_large<T, I, F>(iter: I, f: F)
    where
        I: IntoIterator<Item = T>,
        T: Send,
        F: Fn(T) + Send;

    /// Run many small computations which are independent.
    fn do_many_small<T, I, F>(iter: I, f: F)
    where
        I: IntoIterator<Item = T>,
        T: Send,
        F: Fn(T) + Send;
}

/// Supply to generic functions to execute sequentially (no parallelism).
pub enum Sequential {}

impl ExecutionMethod for Sequential {
    #[inline(always)]
    fn do_both<F, G>(f: F, g: G)
    where
        F: FnOnce() + Send,
        G: FnOnce() + Send,
    {
        f();
        g();
    }

    #[inline(always)]
    fn do_many_large<T, I, F>(iter: I, f: F)
    where
        I: IntoIterator<Item = T>,
        T: Send,
        F: Fn(T) + Send,
    {
        for x in iter {
            f(x);
        }
    }

    #[inline(always)]
    fn do_many_small<T, I, F>(iter: I, f: F)
    where
        I: IntoIterator<Item = T>,
        T: Send,
        F: Fn(T) + Send,
    {
        for x in iter {
            f(x);
        }
    }
}
