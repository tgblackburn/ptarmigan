//! Creating groups within the file structure

#[cfg(feature = "with-mpi")]
use mpi::traits::*;

#[cfg(not(feature = "with-mpi"))]
use no_mpi::*;

use hdf5_sys::{
    h5i,
    h5g,
    h5p,
};

use crate::{
    to_c_string,
    Dataset,
    check,
    OutputError,
};

pub trait GroupHolder<C: Communicator>: Sized {
    fn comm<'a>(&'a self) -> &'a C;

    /// Returns the identifier of the HDF5 group
    fn id(&self) -> h5i::hid_t;

    fn specific_rank(&self) -> Option<i32>;

    /// Sets as the default that all MPI tasks write to datasets created
    /// in this group
    fn all_tasks(self) -> Self;

    /// Sets as the default that only a specific MPI task writes to
    /// datasets created in this group
    fn only_task(self, id: i32) -> Self;

    /// Creates a subgroup within the current group or file,
    /// inheriting any task-related write restrictions from
    /// the parent group
    fn new_group<'a>(&'a self, name: &str) -> Result<Group<'a, C>, OutputError> where C: Communicator {
        let cstr = to_c_string(name)?;

        let id = unsafe {
            check!( h5g::H5Gcreate(
                self.id(),
                cstr.as_ptr(),
                h5p::H5P_DEFAULT,
                h5p::H5P_DEFAULT,
                h5p::H5P_DEFAULT,
            ))?
        };

        Ok(Group {
            comm: self.comm(),
            id,
            // inherit from parent
            specific_rank: self.specific_rank(),
        })
    }

    /// Creates a new dataset handle within the current group or file
    fn new_dataset<'a>(&'a self, name: &str) -> Result<Dataset<'a, Self, C>, OutputError> {
        let name = to_c_string(name)?;
        Ok(Dataset::create_in(&self, name, self.specific_rank()))
    }
}

pub struct Group<'a, C> where C: Communicator {
    comm: &'a C,
    id: h5i::hid_t,
    specific_rank: Option<i32>,
}

impl<'a, C> Drop for Group<'a, C> where C: Communicator {
    fn drop(&mut self) {
        unsafe {
            h5g::H5Gclose(self.id);
        }
    }
}

impl<'a, C> GroupHolder<C> for Group<'a, C> where C: Communicator {
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