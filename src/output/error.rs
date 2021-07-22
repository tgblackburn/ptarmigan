//! Output-related errors

use std::fmt;
use std::error::Error;

pub enum OutputError {
    Conversion(String, String),
    Dimension(usize),
    Write(String),
}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OutputError::Conversion(s, t) => write!(f, "'{}' does not specify a valid {}", s, t),
            OutputError::Dimension(d) => write!(f, "requested dimension was {}, only 1 and 2 are supported", d),
            OutputError::Write(s) => writeln!(f, "failed to write histogram to '{}'", s),
        }
    }
}

impl fmt::Debug for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl Error for OutputError {}
