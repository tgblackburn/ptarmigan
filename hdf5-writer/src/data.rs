//! Writing to and reading from datasets

use std::{ffi, mem::MaybeUninit};

#[cfg(feature = "with-mpi")]
use mpi::{traits::*, collective::*};

#[cfg(not(feature = "with-mpi"))]
use no_mpi::*;

use hdf5_sys::{
    h5,
    h5a,
    h5d,
    h5i,
    h5p,
    h5s,
    h5t,
};

use crate::{
    check,
    to_c_string,
    Dataset,
    DatasetReader,
    GroupHolder,
    OutputError,
    Hdf5Type,
};

/// Part of the dataset specific to this MPI process
pub struct ScatteredDataset<T> {
    /// 1D array of data elements
    pub data: Vec<T>,
    /// Dimensions of the process-specific dataset
    pub dims: Vec<usize>,
}

impl<T> ScatteredDataset<T> {
    /// Extract the data, consuming self.
    pub fn take(self) -> Vec<T> {
        self.data
    }
}

/// Data that can be written to or read from an HDF5 dataset
pub trait Hdf5Data {
    type Output;

    fn write_into<G, C>(&self, ds: &Dataset<G, C>, name: &ffi::CStr) -> Result<Option<h5i::hid_t>, OutputError>
        where G: GroupHolder<C>, C: Communicator;

    fn read_from<C>(ds: &DatasetReader<C>) -> Result<Self::Output, OutputError> where C: Communicator;
}

// Scalar (single value) T
impl<T> Hdf5Data for T where T: Hdf5Type {
    type Output = T;

    fn write_into<G, C>(&self, ds: &Dataset<G, C>, name: &ffi::CStr) -> Result<Option<h5i::hid_t>, OutputError>
        where G: GroupHolder<C>, C: Communicator
    {
        // Get hid of source group
        let parent_id = ds.parent().id();

        // MPI comm info
        let rank = ds.parent().comm().rank();
        let ntasks = ds.parent().comm().size();

        // How many things are we actually writing?
        // Either 1 (single task only, id in range) or ntasks (all)
        let is_scalar = if ds.specific_rank().is_some() {
            if ds.specific_rank().unwrap() >= ntasks {
                return Ok(None)
            } else {
                true
            }
        } else {
            false
        };

        unsafe {
            // What type are we writing?
            let datatype = T::new(); // deallocated at end of scope
            let type_id = datatype.id();

            // Create dataspace for dataset
            let filespace = if is_scalar {
                check!(h5s::H5Screate(h5s::H5S_SCALAR))?
            } else {
                let dims = [ntasks as u64];
                check!( h5s::H5Screate_simple(
                    1,
                    dims.as_ptr(),
                    std::ptr::null()
                ))?
            };

            // Create dataset with default properties and close filespace
            let dset_id = check!( h5d::H5D_create(
                parent_id,
                name.as_ptr(),
                type_id,
                filespace,
                h5p::H5P_DEFAULT,
                h5p::H5P_DEFAULT,
                h5p::H5P_DEFAULT,
            ))?;

            check!(h5s::H5Sclose(filespace))?;

            let (memspace, filespace) = if is_scalar {
                let memspace = h5s::H5S_ALL;
                let filespace = if ds.specific_rank().unwrap() == rank {
                    h5s::H5S_ALL
                } else {
                    let filespace = h5d::H5Dget_space(dset_id);
                    h5s::H5Sselect_none(filespace);
                    filespace
                };
                (memspace, filespace)
            } else {
                // Each process defines dataset in memory and writes
                // to a particular hyperslab
                let dims = [1];
                let memspace = check!( h5s::H5Screate_simple(
                    1,
                    dims.as_ptr(),
                    std::ptr::null()
                ))?;

                // Select hyperslab
                let filespace = h5d::H5Dget_space(dset_id);
                let start = [rank as h5::hsize_t];
                let count = [1];
                check!( h5s::H5Sselect_hyperslab(
                    filespace,
                    h5s::H5S_SELECT_SET,
                    start.as_ptr(),
                    std::ptr::null(),
                    count.as_ptr(),
                    std::ptr::null(),
                ))?;

                (memspace, filespace)
            };

            // Create property list for collective write
            let plist_id = check!(
                h5p::H5Pcreate(*h5p::H5P_CLS_DATASET_XFER)
            )?;

            #[cfg(feature = "with-mpi")]
            check!( h5p::H5Pset_dxpl_mpio(
                plist_id,
                h5p::H5FD_mpio_xfer_t::H5FD_MPIO_COLLECTIVE
            ))?;

            // Write
            let ptr: *const T = self;
            check!( h5d::H5Dwrite(
                dset_id,
                type_id,
                memspace,
                filespace,
                plist_id,
                ptr as *const ffi::c_void,
            ))?;

            if is_scalar {
                // nobody closes memspace == H5S_ALL
                // check!(h5s::H5Sclose(memspace))?;
                if ds.specific_rank().unwrap() != rank {
                    check!(h5s::H5Sclose(filespace))?;
                }
            } else {
                check!(h5s::H5Sclose(memspace))?;
                check!(h5s::H5Sclose(filespace))?;
            }
            check!(h5p::H5Pclose(plist_id))?;

            Ok(Some(dset_id))
        }
    }

    fn read_from<C>(ds: &DatasetReader<C>) -> Result<Self::Output, OutputError> where C: Communicator {
        // First, check that we're reading the right datatype
        let datatype = T::new(); // deallocated at end of scope
        let target_type_id = datatype.id();

        let types_are_equal = unsafe {
            h5t::H5Tequal(ds.type_id(), target_type_id) > 0
        };

        if !types_are_equal {
            let type_name = std::any::type_name::<T>().to_owned();
            return Err(OutputError::TypeMismatch(type_name));
        }

        // Then verify the dataspace is scalar!
        if !ds.is_scalar() {
            let type_name = format!("scalar {}", std::any::type_name::<T>());
            return Err(OutputError::TypeMismatch(type_name));
        }

        let data = unsafe {
            if ds.is_attribute() {
                let mut buffer = MaybeUninit::<T>::uninit();

                check!( h5a::H5Aread(
                    ds.id(),
                    ds.type_id(),
                    buffer.as_mut_ptr() as *mut ffi::c_void,
                ))?;

                buffer.assume_init()
            } else {
                let filespace = check!( h5d::H5Dget_space(ds.id()) )?;
                let memspace = check!( h5s::H5Screate(h5s::H5S_SCALAR))?;

                let mut buffer = MaybeUninit::<T>::uninit();

                check!( h5d::H5Dread(
                    ds.id(),
                    ds.type_id(),
                    memspace,
                    filespace,
                    h5p::H5P_DEFAULT,
                    buffer.as_mut_ptr() as *mut ffi::c_void,
                ))?;

                let data = buffer.assume_init();

                check!( h5s::H5Sclose(memspace) )?;
                check!( h5s::H5Sclose(filespace) )?;

                data
            }
        };

        Ok(data)
    }
}

// Slices &[T]
impl<T> Hdf5Data for [T] where T: Hdf5Type {
    type Output = ScatteredDataset<T>;

    fn write_into<G, C>(&self, ds: &Dataset<G, C>, name: &ffi::CStr) -> Result<Option<h5i::hid_t>, OutputError>
        where G: GroupHolder<C>, C: Communicator
    {
        // Get hid of source group
        let parent_id = ds.parent().id();

        // How many things are we actually writing?
        let rank = ds.parent().comm().rank();
        let ntasks = ds.parent().comm().size();

        let len = self.len() as h5::hsize_t;

        let (count, local_count, local_offset) = if ds.specific_rank().is_some() {
            let specific_rank = ds.specific_rank().unwrap();

            // If requested rank is outside the communicator, do nothing
            if specific_rank >= ntasks {
                return Ok(None)
            }

            // Everyone has to agree on how large a dataset to create
            let mut counts = vec![0 as h5::hsize_t; ntasks as usize];
            ds.parent().comm().all_gather_into(&len, &mut counts[..]);

            let count = counts[specific_rank as usize];

            if specific_rank == rank {
                (count, count, 0)
            } else {
                (count, 0, 0)
            }
        } else { // everybody writing
            let mut counts = vec![0 as h5::hsize_t; ntasks as usize];
            // initially, counts = [0, 0, 0, 0, ...]
            ds.parent().comm().all_gather_into(&len, &mut counts[..]);
            // now counts = [n_0, n_1, n_2, ...]
            let count: h5::hsize_t = counts.iter().sum();

            // Determine offsets
            let mut offsets = counts.clone();
            offsets.iter_mut().fold(0, |mut total, n| {
                total += *n;
                *n = total - *n;
                total
            });
            // offsets = [0, n_0, n_0 + n_1, ...]

            (count, counts[rank as usize], offsets[rank as usize])
        };

        let empty_dset = count == 0;

        unsafe {
            // What type are we writing?
            let datatype = T::new(); // deallocated at end of scope
            let type_id = datatype.id();

            // Create dataspace for dataset
            let dims = [count];
            let filespace = check!( h5s::H5Screate_simple(
                1,
                dims.as_ptr(),
                std::ptr::null(),
            ))?;

            // Create dataset with default properties and close filespace
            let dset_id = check!( h5d::H5D_create(
                parent_id,
                name.as_ptr(),
                type_id,
                filespace,
                h5p::H5P_DEFAULT,
                h5p::H5P_DEFAULT,
                h5p::H5P_DEFAULT,
            ))?;

            check!(h5s::H5Sclose(filespace))?;

            // Each process defines dataset in memory and writes
            // to a particular hyperslab
            let dims = [local_count];
            let memspace = check!( h5s::H5Screate_simple(
                1,
                dims.as_ptr(),
                std::ptr::null()
            ))?;

            // Select hyperslab
            let filespace = h5d::H5Dget_space(dset_id);
            let start = [local_offset];
            let count = [local_count];
            check!( h5s::H5Sselect_hyperslab(
                filespace,
                h5s::H5S_SELECT_SET,
                start.as_ptr(),
                std::ptr::null(),
                count.as_ptr(),
                std::ptr::null(),
            ))?;

            // Create property list for collective write
            let plist_id = check!(h5p::H5Pcreate(*h5p::H5P_CLS_DATASET_XFER))?;

            #[cfg(feature = "with-mpi")]
            check!( h5p::H5Pset_dxpl_mpio(
                plist_id,
                h5p::H5FD_mpio_xfer_t::H5FD_MPIO_COLLECTIVE
            ))?;

            // Write
            if !empty_dset {
                check!( h5d::H5Dwrite(
                    dset_id,
                    type_id,
                    memspace,
                    filespace,
                    plist_id,
                    self.as_ptr() as *const ffi::c_void,
                ))?;
            }

            check!(h5s::H5Sclose(filespace))?;
            check!(h5s::H5Sclose(memspace))?;
            check!(h5p::H5Pclose(plist_id))?;

            Ok(Some(dset_id))
        }
    }

    fn read_from<C>(ds: &DatasetReader<C>) -> Result<Self::Output, OutputError> where C: Communicator {
        // First, check that we're reading the right datatype
        let datatype = T::new(); // deallocated at end of scope
        let target_type_id = datatype.id();

        let types_are_equal = unsafe {
            h5t::H5Tequal(ds.type_id(), target_type_id) > 0
        };

        if !types_are_equal {
            let type_name = std::any::type_name::<T>().to_owned();
            return Err(OutputError::TypeMismatch(type_name));
        }

        // What if we have a scalar dataset?
        if ds.dims().is_empty() {
            let data = if ds.comm().rank() == 0 {
                // only root gets a value
                T::read_from(ds)
                    .map(|v| ScatteredDataset { data: vec![v], dims: vec![1] } )
            } else {
                Ok(ScatteredDataset { data: vec![], dims: vec![] })
            };
            return data;
        }

        // The dataset is divided along the slowest varying dimension.
        let id = ds.comm().rank();
        let counts: Vec<usize> = {
            let npart = ds.dims()[0];
            let tasks = ds.comm().size() as usize;
            (0..tasks).map(|i| (npart * (i + 1) / tasks) - (npart * i / tasks)).collect()
        };
        let count = counts[id as usize];

        let mut offset = 0;
        for count in counts.iter().take(id as usize) {
            offset += count;
        };
        let offset = offset;

        // println!("\tscatter: {} reading {} at offset {}", self.comm.rank(), count, offset);

        let data = unsafe {
            let filespace = check!( h5d::H5Dget_space(ds.id()) )?;

            let ndims = ds.dims().len();

            // offset of the block from the start of the dimension
            let mut start: Vec<h5::hsize_t> = vec![0; ndims];
            start[0] = offset as h5::hsize_t;

            // how many blocks there are in each dimension = only one
            let counts: Vec<h5::hsize_t> = vec![1; ndims];

            // block[0] = count, i.e. length of dataset / MPI processes
            // all other block[i] are the complete size of the dataset
            let mut block: Vec<_> = ds.dims().iter().map(|i| *i as h5::hsize_t).collect();
            block[0] = count as h5::hsize_t;

            check!( h5s::H5Sselect_hyperslab(
                filespace,
                h5s::H5S_SELECT_SET,
                start.as_ptr(),
                std::ptr::null(), // take every element
                counts.as_ptr(),
                block.as_ptr(),
            ))?;

            let nelems: h5::hsize_t = block.iter().product();
            // let nelems = check!( h5s::H5Sget_select_npoints(filespace) )?;

            // writing to this thing
            let dims = [nelems];
            let memspace = check!( h5s::H5Screate_simple(
                1,
                dims.as_ptr(),
                std::ptr::null()
            ))?;

            let mut buffer: Vec<T> = Vec::with_capacity(nelems as usize);

            check!( h5d::H5Dread(
                ds.id(),
                ds.type_id(),
                memspace,
                filespace,
                h5p::H5P_DEFAULT,
                buffer.as_mut_ptr() as *mut ffi::c_void,
            ))?;

            buffer.set_len(nelems as usize);

            check!( h5s::H5Sclose(memspace) )?;
            check!( h5s::H5Sclose(filespace) )?;

            let dims: Vec<usize> = block.iter().map(|n| *n as usize).collect();

            ScatteredDataset { data: buffer, dims }
        };

        Ok(data)
    }
}

// Strings and string slices
impl Hdf5Data for String {
    type Output = String;

    fn write_into<G, C>(&self, ds: &Dataset<G, C>, name: &ffi::CStr) -> Result<Option<h5i::hid_t>, OutputError>
        where G: GroupHolder<C>, C: Communicator
    {
        self.as_str().write_into(ds, name)
    }

    fn read_from<C>(ds: &DatasetReader<C>) -> Result<Self::Output, OutputError> where C: Communicator {
        str::read_from(ds)
    }
}

impl Hdf5Data for str {
    type Output = String;

    fn write_into<G, C>(&self, ds: &Dataset<G, C>, name: &ffi::CStr) -> Result<Option<h5i::hid_t>, OutputError>
        where G: GroupHolder<C>, C: Communicator
    {
        // Get hid of source group
        let parent_id = ds.parent().id();

        // How many processes are actually writing?
        let rank = ds.parent().comm().rank();
        let ntasks = ds.parent().comm().size();

        let (is_scalar, offset) = if ds.specific_rank().is_some() {
            // If requested rank is outside the communicator, do nothing
            if ds.specific_rank().unwrap() >= ntasks {
                return Ok(None)
            }
            (true, 0_u64)
        } else {
            (false, rank as u64)
        };

        // Turn string into something sensible!
        let data = to_c_string(self.as_ref())?;

        // MPI doesn't like variable-length arrays
        let local_len = data.to_bytes().len();

        // if scalar, we only need the string being written by the
        // relevant rank. otherwise we need the largest value of
        // all possible strings.
        let len = if is_scalar {
            let origin = ds.specific_rank().unwrap();
            // This doesn't work unless origin = 0:
            // let mut len = local_len;
            // let root = ds.parent().comm().process_at_rank(origin);
            // root.broadcast_into(&mut len);
            let mut lens = vec![0_usize; ntasks as usize];
            ds.parent().comm().all_gather_into(&local_len, &mut lens[..]);
            lens[origin as usize]
        } else {
            let mut len = local_len;
            ds.parent().comm()
                .all_reduce_into(
                    &local_len,
                    &mut len,
                    SystemOperation::max()
                );
            len
        };

        unsafe {
            let type_id = check!(h5t::H5Tcopy(*h5t::H5T_C_S1))?;
            check!(h5t::H5Tset_size(type_id, len))?;
            // This won't work:
            // h5t::H5Tset_size(type_id, h5t::H5T_VARIABLE).check()?;
            check!(h5t::H5Tset_strpad(type_id, h5t::H5T_STR_NULLTERM))?;
            check!(h5t::H5Tset_cset(type_id, h5t::H5T_CSET_UTF8))?;

            // Create dataspace for dataset
            let filespace = if is_scalar {
                check!(h5s::H5Screate(h5s::H5S_SCALAR))?
            } else {
                let dims = [ntasks as u64];
                check!( h5s::H5Screate_simple(
                    1,
                    dims.as_ptr(),
                    std::ptr::null()
                ))?
            };

            // Create dataset with default properties and close filespace
            let dset_id = check!( h5d::H5D_create(
                parent_id,
                name.as_ptr(),
                type_id,
                filespace,
                h5p::H5P_DEFAULT,
                h5p::H5P_DEFAULT,
                h5p::H5P_DEFAULT,
            ))?;

            check!(h5s::H5Sclose(filespace))?;

            let (memspace, filespace) = if is_scalar {
                let memspace = h5s::H5S_ALL;
                let filespace = if ds.specific_rank().unwrap() == rank {
                    h5s::H5S_ALL
                } else {
                    let filespace = h5d::H5Dget_space(dset_id);
                    h5s::H5Sselect_none(filespace);
                    filespace
                };
                (memspace, filespace)
            } else {
                // Each process defines dataset in memory and writes
                // to a particular hyperslab
                let dims = [1];
                let memspace = check!( h5s::H5Screate_simple(
                    1,
                    dims.as_ptr(),
                    std::ptr::null()
                ))?;

                // Select hyperslab
                let filespace = h5d::H5Dget_space(dset_id);
                let start = [offset];
                let count = [1];
                check!( h5s::H5Sselect_hyperslab(
                    filespace,
                    h5s::H5S_SELECT_SET,
                    start.as_ptr(),
                    std::ptr::null(),
                    count.as_ptr(),
                    std::ptr::null(),
                ))?;

                (memspace, filespace)
            };

            // Create property list for collective write
            let plist_id = check!(
                h5p::H5Pcreate(*h5p::H5P_CLS_DATASET_XFER)
            )?;

            #[cfg(feature = "with-mpi")]
            check!( h5p::H5Pset_dxpl_mpio(
                plist_id,
                h5p::H5FD_mpio_xfer_t::H5FD_MPIO_COLLECTIVE
            ))?;

            // Write
            check!( h5d::H5Dwrite(
                dset_id,
                type_id,
                memspace,
                filespace,
                plist_id,
                data.as_ptr() as *const ffi::c_void,
            ))?;

            // check!(h5d::H5Dclose(dset_id))?;
            if is_scalar {
                // nobody closes memspace == H5S_ALL
                // check!(h5s::H5Sclose(memspace))?;
                if ds.specific_rank().unwrap() != rank {
                    check!(h5s::H5Sclose(filespace))?;
                }
            } else {
                check!(h5s::H5Sclose(memspace))?;
                check!(h5s::H5Sclose(filespace))?;
            }
            check!(h5p::H5Pclose(plist_id))?;

            Ok(Some(dset_id))
        }
    }

    fn read_from<C>(ds: &DatasetReader<C>) -> Result<Self::Output, OutputError> where C: Communicator {
        // What kind of string do we have?
        let is_string = unsafe {
            let class = h5t::H5Tget_class(ds.type_id());
            class == h5t::H5T_STRING
        };

        if !ds.is_scalar() || !is_string {
            let type_name = "scalar string".to_owned();
            return Err(OutputError::TypeMismatch(type_name));
        }

        unsafe {
            let len = h5t::H5Tget_size(ds.type_id()); // not including null terminator
            let mut buffer: Vec<u8> = vec![0; len];

            if ds.is_attribute() {
                check!( h5a::H5Aread(
                    ds.id(),
                    ds.type_id(),
                    buffer.as_mut_ptr() as *mut ffi::c_void,
                ))?;
            } else {
                let filespace = check!( h5d::H5Dget_space(ds.id()) )?;
                let memspace = check!( h5s::H5Screate(h5s::H5S_SCALAR))?;

                check!( h5d::H5Dread(
                    ds.id(),
                    ds.type_id(),
                    memspace,
                    filespace,
                    h5p::H5P_DEFAULT,
                    buffer.as_mut_ptr() as *mut ffi::c_void,
                ))?;

                check!( h5s::H5Sclose(memspace) )?;
                check!( h5s::H5Sclose(filespace) )?;
            }

            String::from_utf8(buffer).map_err(|_| OutputError::TypeMismatch("valid UTF-8".to_owned()))
        }
    }
}