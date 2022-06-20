//! Writing output to HDF5, in serial or parallel via MPI

use std::ffi;

// #[cfg(feature = "with-mpi")]
// use mpi::traits::*;

// #[cfg(not(feature = "with-mpi"))]
// mod no_mpi;
// #[cfg(not(feature = "with-mpi"))]
// use no_mpi::*;
// #[cfg(not(feature = "with-mpi"))]
// use no_mpi as mpi;

mod dataset;
mod datatype;
mod error;
mod file;
mod group;
mod write;

pub use dataset::*;
pub use datatype::*;
pub use error::*;
pub use file::*;
pub use group::*;
use write::*;

/// Copies a Rust string slice to a owned, nul-terminated C string
fn to_c_string(str: &str) -> Result<ffi::CString, OutputError> {
    ffi::CString::new(str).or_else(|_e|
        Err(OutputError::Identifier(str.to_owned()))
    )
}
