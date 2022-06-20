//! Error handling

use std::error::Error;
use std::fmt;

// use libc_stdhandle;

use hdf5_sys::{
    h5,
    h5i,
};


/// Wraps a call to an libhdf5 function, returning a Result which is either
/// Ok(return value) or Err(OutputError).
#[macro_export]
macro_rules! check {
    ($class:ident::$func:ident($($args:expr),* $(,)?)) => {{
        use crate::{OutputError, Checkable};
        // invoke function
        let val = $class::$func($($args,)*);
        if val.is_error_code() {
            // #[allow(unused_unsafe)]
            // unsafe {
            //     use hdf5_sys::h5e;
            //     // Has to be called here, otherwise error unwinding starts to free libhdf5
            //     // resources, which succeed and clear the error stack
            //     h5e::H5Eprint(h5e::H5E_DEFAULT, libc_stdhandle::stderr());
            // };
            Err(OutputError::H5Call(stringify!($func).to_owned(), file!().to_owned(), line!()))
        } else {
            Ok(val)
        }
    }}
}

pub enum OutputError {
    Identifier(String),
    H5Call(String, String, u32),
}

impl fmt::Debug for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OutputError::Identifier(s) => {
                write!(f, "Unable to convert requested identifier '{}' to nul-terminated string!", s)
            },
            OutputError::H5Call(func, file, line) => {
                write!(f, "{} (line {} in {}) failed, see diagnostic messages above.", func, line, file)
            }
        }
    }
}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for OutputError {}

pub trait Checkable: Copy {
    fn is_error_code(&self) -> bool;
}

impl Checkable for h5i::hid_t {
    fn is_error_code(&self) -> bool {
        *self < 0
    }
}

impl Checkable for h5::herr_t {
    fn is_error_code(&self) -> bool {
        *self < 0
    }
}