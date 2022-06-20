//! Creating HDF5 files in parallel

#[cfg(feature = "with-mpi")]
use {
    std::mem::MaybeUninit,
    mpi::traits::*,
    mpi_sys,
};

#[cfg(not(feature = "with-mpi"))]
use no_mpi::*;

use hdf5_sys::{
    h5e,
    h5f,
    h5i,
    h5p,
};

use crate::{
    to_c_string,
    GroupHolder,
};

use crate::{
    check,
    OutputError,
};

/// Represents an open HDF5 file that can be written to in parallel.
pub struct ParallelFile<'a, C> where C: Communicator {
    comm: &'a C,
    id: h5i::hid_t,
    specific_rank: Option<i32>,
}

impl<'a, C> ParallelFile<'a, C> where C: Communicator {
    /// Opens a new HDF5 file handle, in parallel.
    /// All future operations on this file handle must be executed by
    /// *all processes* in the communicator that opened it.
    /// ```
    /// let universe = mpi::initialize().unwrap();
    /// let world = universe.world();
    /// let file = ParallelFile::create(&world, "test.h5").unwrap();
    /// ...
    /// // Must be called by all processes in `world`:
    /// file.new_group("group").unwrap();
    /// ```
    pub fn create(comm: &'a C, filename: &str) -> Result<Self, OutputError> {
        let filename = to_c_string(filename)?;

        let id = unsafe {
            // Silence errors
            check!( h5e::H5Eset_auto(
                h5e::H5E_DEFAULT,
                None,
                std::ptr::null_mut()
            ))?;

            // Set up file access property list with parallel IO
            let plist = check!(
                h5p::H5Pcreate(*h5p::H5P_CLS_FILE_ACCESS)
            )?;

            #[cfg(feature = "with-mpi")] {
                // Get MPI_INFO_NULL
                let info: mpi_sys::MPI_Info = {
                    let mut info: MaybeUninit<_> = MaybeUninit::uninit();
                    mpi_sys::MPI_Info_create(info.as_mut_ptr());
                    let mut info = info.assume_init();
                    mpi_sys::MPI_Info_free(&mut info);
                    info
                };

                check!(h5p::H5Pset_fapl_mpio(plist, comm.as_raw(), info))?;
            }

            // Collectively create a new file
            let id = check!( h5f::H5Fcreate(
                filename.as_ptr(),
                h5f::H5F_ACC_TRUNC,
                h5p::H5P_DEFAULT,
                plist
            ))?;

            // Close property list
            check!(h5p::H5Pclose(plist))?;

            id
        };

        Ok(Self {
            comm,
            id,
            specific_rank: None,
        })
    }
}

impl<'a, C> Drop for ParallelFile<'a, C> where C: Communicator {
    fn drop(&mut self) {
        unsafe {
            // Close the file
            check!(h5f::H5Fclose(self.id))
                .expect("Failed to close the file");
        }
    }
}

impl<'a, C> GroupHolder<C> for ParallelFile<'a, C> where C: Communicator {
    fn comm(&self) -> &C where C: Communicator {
        self.comm
    }

    fn id(&self) -> h5i::hid_t {
        self.id
    }

    fn specific_rank(&self) -> Option<i32> {
        self.specific_rank
    }

    fn all_tasks(mut self) -> Self {
        self.specific_rank = None;
        self
    }

    fn only_task(mut self, id: i32) -> Self {
        self.specific_rank = Some(id);
        self
    }
}