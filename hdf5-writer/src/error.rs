//! Error handling

use std::error::Error;
use std::fmt;
use std::ffi::CStr;
use libc;

use hdf5_sys::{
    h5,
    h5e,
};

/// Wraps a call to an libhdf5 function, returning a Result which is either
/// Ok(return value) or Err(OutputError).
#[macro_export]
macro_rules! check {
    ($class:ident::$func:ident($($args:expr),* $(,)?)) => {{
        use crate::{OutputError, Checkable, print_error_stack};
        // invoke function
        let val = $class::$func($($args,)*);
        if val.is_error_code() {
            use hdf5_sys::h5e;
            #[allow(unused_unsafe)]
            unsafe {
                h5e::H5Ewalk(
                    h5e::H5E_DEFAULT,
                    h5e::H5E_WALK_DOWNWARD,
                    Some(print_error_stack),
                    std::ptr::null_mut() as *mut libc::c_void,
                );
            }
            Err(OutputError::H5Call {
                func: stringify!($func).to_owned(),
                file: file!().to_owned(),
                line: line!(),
            })
        } else {
            Ok(val)
        }
    }}
}

// Callback that prints the error stack, which can be passed to HDF5 library
#[allow(deprecated)]
pub unsafe extern "C" fn print_error_stack(n: libc::c_uint, error_desc: *const h5e::H5E_error_t, _: *mut libc::c_void) -> h5::herr_t {
    if error_desc.is_null() {
        return -1;
    }

    let file_name = CStr::from_ptr((*error_desc).file_name);
    let line_num = (*error_desc).line;

    let func_name = CStr::from_ptr((*error_desc).func_name);
    let desc = CStr::from_ptr((*error_desc).desc);

    let maj: h5e::H5E_major_t = (*error_desc).maj_num;
    let min: h5e::H5E_minor_t = (*error_desc).min_num;
    let maj_str = CStr::from_ptr(h5e::H5Eget_major(maj));
    let min_str = CStr::from_ptr(h5e::H5Eget_minor(min));

    eprintln!(
        "#{:<03}: {} line {} in {}(): {}\n  major: {}\n  minor: {}",
        n, file_name.to_str().unwrap(), line_num, func_name.to_str().unwrap(), desc.to_str().unwrap(),
        maj_str.to_str().unwrap(), min_str.to_str().unwrap(),
    );

    0
}

pub enum OutputError {
    Identifier(String),
    H5Call {
        func: String,
        file: String,
        line: u32,
    },
}

impl fmt::Debug for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OutputError::Identifier(s) => {
                write!(f, "Unable to convert requested identifier '{}' to nul-terminated string!", s)
            },
            OutputError::H5Call {func, file, line} => {
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

pub trait Checkable {
    fn is_error_code(self) -> bool;
}

impl<T> Checkable for T where T: Copy + From<i32> + Ord {
    fn is_error_code(self) -> bool {
        self < 0.into()
    }
}