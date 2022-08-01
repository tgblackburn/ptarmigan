//! Writing to datasets

use std::ffi;

#[cfg(feature = "with-mpi")]
use mpi::{traits::*, collective::*};

#[cfg(not(feature = "with-mpi"))]
use no_mpi::*;

use hdf5_sys::{
    h5,
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
    GroupHolder,
    OutputError,
    Hdf5Type,
};

/// Data that can be written to an HDF5 dataset
pub trait Hdf5Data {
    fn write_into<G, C>(&self, ds: &Dataset<G, C>, name: &ffi::CStr) -> Result<Option<h5i::hid_t>, OutputError>
        where G: GroupHolder<C>, C: Communicator;
}

// Writing of Vec<T> by coercing to slice
// impl<V, T> Hdf5Data for V where V: Deref<Target=[T]>, T: Hdf5Type {
//     fn write_into<G, C>(&self, ds: &Dataset<G, C>, name: &ffi::CStr) -> Result<Option<h5i::hid_t>, OutputError>
//             where G: GroupHolder<C>, C: Communicator {
//         <[T]>::write_into(&self, ds, name)
//     }
// }

// Writing of scalar (single value) T
impl<T> Hdf5Data for T where T: Hdf5Type {
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
}

// Writing of slices &[T]
impl<T> Hdf5Data for [T] where T: Hdf5Type {
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
}

// Writing of strings and string slices
impl Hdf5Data for String {
    fn write_into<G, C>(&self, ds: &Dataset<G, C>, name: &ffi::CStr) -> Result<Option<h5i::hid_t>, OutputError>
        where G: GroupHolder<C>, C: Communicator
    {
        self.as_str().write_into(ds, name)
    }
}

impl Hdf5Data for str {
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
}