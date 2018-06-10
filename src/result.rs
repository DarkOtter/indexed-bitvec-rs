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
use std::result;
use std::error;
use std::fmt;

#[derive(Clone, Copy, Debug)]
pub enum Error {
    IndexIncorrectSize,
}

pub type Result<T> = result::Result<T, Error>;

impl error::Error for Error {
    fn description(&self) -> &str {
        use self::Error::*;
        match self {
            IndexIncorrectSize => "Index is the wrong size for the bitvector used",
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, into: &mut fmt::Formatter) -> fmt::Result {
        use self::error::Error;

        self.description().fmt(into)
    }
}
