//! Creating datasets

use std::marker::PhantomData;
use std::ffi;

#[cfg(feature = "with-mpi")]
use mpi::{traits::*, collective::*};

#[cfg(not(feature = "with-mpi"))]
use no_mpi::*;

use hdf5_sys::{
    h5a,
    h5d,
    h5i,
    h5l,
    h5p,
    h5s,
    h5t,
};

use crate::{
    to_c_string,
    GroupHolder,
    OutputError,
    Hdf5Data,
    check,
};

pub struct Dataset<'a, G, C> where G: GroupHolder<C>, C: Communicator {
    parent: &'a G,
    name: ffi::CString,
    unit: Option<ffi::CString>,
    desc: Option<ffi::CString>,
    condition: bool,
    specific_rank: Option<i32>,
    aliases: Vec<ffi::CString>,
    pd: PhantomData<C>,
}

impl<'a, G, C> Dataset<'a, G, C> where G: GroupHolder<C>, C: Communicator {
    /// Creates an empty dataset inside the specified group (or file)
    pub fn create_in(parent: &'a G, name: ffi::CString, specific_rank: Option<i32>) -> Self {
        Self {
            parent: parent,
            name: name,
            unit: None,
            desc: None,
            condition: true,
            specific_rank,
            aliases: vec![],
            pd: PhantomData::<C>,
        }
    }

    /// Parent group of this dataset
    pub fn parent(&self) -> &'a G {
        self.parent
    }

    pub fn specific_rank(&self) -> Option<i32> {
        self.specific_rank
    }

    /// Assign a unit to the data output.
    /// This can only fail if the argument cannot be converted
    /// to a C-style string, i.e. it contains NUL bytes.
    pub fn with_unit(mut self, unit: &str) -> Result<Self, OutputError> {
        let unit = to_c_string(unit)?;
        self.unit = Some(unit);
        Ok(self)
    }

    /// Assign a description of the dataset.
    /// This can only fail if the argument cannot be converted
    /// to a C-style string, i.e. it contains NUL bytes.
    pub fn with_desc(mut self, desc: &str) -> Result<Self, OutputError> {
        let desc = to_c_string(desc)?;
        self.desc = Some(desc);
        Ok(self)
    }

    /// Means that the dataset will only be written if the closure returns True
    /// on *all participating processes*
    #[allow(unused)]
    pub fn with_condition<F>(mut self, f: F) -> Self where F: FnOnce() -> bool {
        self.condition = f();
        self
    }

    /// Means that all processes write data to the specified dataset
    #[allow(unused)]
    pub fn all_tasks(mut self) -> Self {
        self.specific_rank = None;
        self
    }

    /// Ensures that a future [write](Dataset::write) operation on this dataset will only
    /// be executed by the task with the given ID.
    /// All tasks must still call [write](Dataset::write) to avoid blocking
    pub fn only_task(mut self, id: i32) -> Self {
        self.specific_rank = Some(id);
        self
    }

    /// Adds an alternative name for this dataset by creating an appropriate
    /// soft link
    pub fn with_alias(mut self, alias: &str) -> Result<Self, OutputError> {
        let alias = to_c_string(alias)?;
        self.aliases.push(alias);
        Ok(self)
    }

    unsafe fn attach_attribute(dataset: h5i::hid_t, name: &str, value: &ffi::CStr) -> Result<(), OutputError> {
        // Create dataspace for attribute
        let space_id = check!(h5s::H5Screate(h5s::H5S_SCALAR))?;

        // Create type
        let type_id = check!(h5t::H5Tcopy(*h5t::H5T_C_S1) )?;
        check!(h5t::H5Tset_size(type_id, value.to_bytes().len()))?;
        check!(h5t::H5Tset_strpad(type_id, h5t::H5T_STR_NULLTERM))?;
        check!(h5t::H5Tset_cset(type_id, h5t::H5T_CSET_UTF8))?;

        // Create the attribute itself
        let name = to_c_string(name)?;
        let attr_id = check!( h5a::H5Acreate(
            dataset,
            name.as_ptr(),
            type_id,
            space_id,
            h5p::H5P_DEFAULT,
            h5p::H5P_DEFAULT,
        ))?;

        // Write the data
        check!( h5a::H5Awrite(
            attr_id,
            type_id,
            value.as_ptr() as *const ffi::c_void
        ))?;

        // Close stuff
        check!(h5a::H5Aclose(attr_id))?;
        check!(h5s::H5Sclose(space_id))?;
        check!(h5t::H5Tclose(type_id))?;

        // Otherwise, all good
        Ok(())
    }

    /// Writes data (a scalar value `&T`, slice `&[T]` or a string slice `&str`) to current
    /// dataset handle, concatenating the data from each MPI task in rank order.
    /// If only a single task writes scalar data, the output will also be scalar.
    /// Returns the group handle so that further datasets can be created.
    pub fn write<T>(self, data: &T) -> Result<&'a G, OutputError> where T: Hdf5Data + ?Sized {
        // Write only if every process agrees
        let mut writing = true;
        self.parent.comm()
            .all_reduce_into(
                &self.condition,
                &mut writing,
                SystemOperation::logical_and()
            );

        if writing {
            let dset_id = data.write_into(&self, self.name.as_ref())?;

            unsafe {
                for alias in self.aliases {
                    check!(h5l::H5Lcreate_soft(
                        self.name.as_ptr(),
                        self.parent.id(),
                        alias.as_ptr(),
                        h5p::H5P_DEFAULT,
                        h5p::H5P_DEFAULT
                    ))?;
                }

                if self.unit.is_some() && dset_id.is_some() {
                    Self::attach_attribute(dset_id.unwrap(), "unit", self.unit.unwrap().as_ref())?;
                }

                if self.desc.is_some() && dset_id.is_some() {
                    Self::attach_attribute(dset_id.unwrap(), "desc", self.desc.unwrap().as_ref())?;
                }

                if dset_id.is_some() {
                    check!(h5d::H5Dclose(dset_id.unwrap()))?;
                }
            }
        }

        Ok(self.parent)
    }
}

pub struct DatasetReader<'a, C> where C: Communicator {
    comm: &'a C,
    id: h5i::hid_t,
    type_id: h5i::hid_t,
    dims: Vec<usize>,
    is_attribute: bool,
}

impl<'a, C> DatasetReader<'a, C> where C: Communicator {
    pub fn open_in<G>(parent: &'a G, name: ffi::CString) -> Result<Self, OutputError> where G: GroupHolder<C> {
        let id = unsafe {
            check!( h5d::H5Dopen(
                parent.id(),
                name.as_ptr(),
                h5p::H5P_DEFAULT,
            ))?
        };

        let type_id = unsafe {
            check!( h5d::H5Dget_type(id) )?
        };

        // Get information about the dataset's dimensions.
        // A scalar dataset has zero rank.
        let dims = unsafe {
            let space_id = check!( h5d::H5Dget_space(id) )?;
            let ndims = check!( h5s::H5Sget_simple_extent_ndims(space_id) )?;

            let dims = if ndims == 0 { // scalar
                vec![]
            } else {
                let mut dims = vec![0; ndims as usize];
                let mut maxdims = vec![0; ndims as usize];
                check!( h5s::H5Sget_simple_extent_dims(
                    space_id,
                    dims.as_mut_ptr(),
                    maxdims.as_mut_ptr()
                ))?;
                dims
            };

            h5s::H5Sclose(space_id);
            dims
        };

        let dims: Vec<_> = dims.into_iter().map(|n| n as usize).collect();

        // println!("\tdataset open: {} got id {}, type {}", parent.comm().rank(), id, type_id);

        Ok(Self {
            comm: parent.comm(),
            id,
            type_id,
            dims,
            is_attribute: false,
        })
    }

    pub fn comm(&self) -> &C {
        &self.comm
    }

    pub fn id(&self) -> h5i::hid_t {
        self.id
    }

    pub fn type_id(&self) -> h5i::hid_t {
        self.type_id
    }

    /// Returns the size of the dataset along each dimension.
    /// If the dataset is scalar, i.e. it contains only a single value,
    /// the slice is empty.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    pub fn is_attribute(&self) -> bool {
        self.is_attribute
    }

    /// Returns the fully qualified path associated with this dataset.
    pub fn path(&self) -> String {
        // length is not known in advance
        let len = unsafe {
            h5i::H5Iget_name(
                self.id,
                std::ptr::null_mut(),
                0
            )
        };

        let len = len.max(0) as usize;

        let str = unsafe {
            // appends null terminator for us
            let buffer = ffi::CString::from_vec_unchecked(vec![0; len])
                .into_raw();

            // read name into buffer
            h5i::H5Iget_name(
                self.id,
                buffer,
                len + 1
            );

            ffi::CString::from_raw(buffer)
        };

        str.into_string().unwrap_or_else(|_| String::new())
    }

    /// Reads the dataset and
    /// * broadcasts a single value to all processes, if `T` is a scalar (e.g. `f64`), or
    /// * divides the contents among all participating MPI processes, if `T` is a slice (e.g. `[f64]`).
    pub fn read<T>(&self) -> Result<<T as Hdf5Data>::Output, OutputError> where T: Hdf5Data + ?Sized {
        <T as Hdf5Data>::read_from(self)
    }

    /// Opens the attribute of the given name, which is attached to this dataset.
    pub fn open_attribute(&'a self, name: &str) -> Result<Self, OutputError> {
        let name = to_c_string(name)?;

        let id = unsafe {
            check!( h5a::H5Aopen(
                self.id(),
                name.as_ptr(),
                h5p::H5P_DEFAULT,
            ))?
        };

        let type_id = unsafe {
            check!( h5a::H5Aget_type(id) )?
        };

        // Get information about the dataset's dimensions.
        // A scalar dataset has zero rank.
        let dims = unsafe {
            let space_id = check!( h5a::H5Aget_space(id) )?;
            let ndims = check!( h5s::H5Sget_simple_extent_ndims(space_id) )?;

            let dims = if ndims == 0 { // scalar
                vec![]
            } else {
                let mut dims = vec![0; ndims as usize];
                let mut maxdims = vec![0; ndims as usize];
                check!( h5s::H5Sget_simple_extent_dims(
                    space_id,
                    dims.as_mut_ptr(),
                    maxdims.as_mut_ptr()
                ))?;
                dims
            };

            h5s::H5Sclose(space_id);
            dims
        };

        let dims: Vec<_> = dims.into_iter().map(|n| n as usize).collect();

        Ok(Self {
            comm: self.comm(),
            id,
            type_id,
            dims,
            is_attribute: true,
        })
    }
}

impl<'a, C> Drop for DatasetReader<'a, C> where C: Communicator {
    fn drop(&mut self) {
        unsafe {
            h5t::H5Tclose(self.type_id);
            if self.is_attribute {
                h5a::H5Aclose(self.id);
            } else {
                h5d::H5Dclose(self.id);
            }
        }
    }
}
