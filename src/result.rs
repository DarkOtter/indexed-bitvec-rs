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
        use self::fmt::Display;

        self.description().fmt(into)
    }
}
