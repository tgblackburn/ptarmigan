//! HDF5-interpretable datatypes

use std::ffi;

use hdf5_sys::{
    h5,
    h5i,
    h5t,
};

use crate::{
    check,
    to_c_string,
};

pub struct Datatype {
    builtin: bool,
    id: h5i::hid_t,
}

impl Datatype {
    pub fn id(&self) -> h5i::hid_t {
        self.id
    }

    /// Construct an HDF5-interpretable enumeration datatype
    pub fn enumeration(src: &[(&str, i32)]) -> Self {
        let id = unsafe {
            let type_id = check!( h5t::H5Tenum_create(i32::new().id()) ).unwrap();
            for (name, value) in src.iter() {
                let name = to_c_string(name).unwrap();
                let value: *const i32 = value;
                check!( h5t::H5Tenum_insert(type_id, name.as_ptr(), value as *const ffi::c_void) ).unwrap();
            }
            type_id
        };
        Datatype { builtin: false, id }
    }
}

impl Drop for Datatype {
    fn drop(&mut self) {
        if !self.builtin {
            unsafe {
                // cannot check for errors here without panicking
                check!(h5t::H5Tclose(self.id))
                    .expect("Failed to close HDF5 datatype");
            }
        }
    }
}

pub trait Hdf5Type {
    /// Construct and register a new HDF5-interpretable datatype
    fn new() -> Datatype;

    /// Construct and register an HDF5 datatype which is an array
    /// of data elements of given length
    fn array(len: i32) -> Datatype {
        let dims = [len as h5::hsize_t];
        let base_type = <Self as Hdf5Type>::new();
        let base_id = base_type.id();
        let id = unsafe {
            check!(h5t::H5Tarray_create(base_id, 1, dims.as_ptr()))
                .unwrap_or_else(|_|
                    panic!("Failed to build HDF5 datatype for [{}; {}]", std::any::type_name::<Self>(), len)
                )
        };
        Datatype {builtin: false, id}
    }
}

macro_rules! impl_hdf5type_for_builtin {
    ($type:ident, $le:ident, $be:ident) => {
        impl Hdf5Type for $type {
            #[cfg(target_endian = "little")]
            fn new()-> Datatype {
                Datatype { builtin: true, id: *h5t::$le }
            }

            #[cfg(target_endian = "big")]
            fn new()-> Datatype {
                Datatype { builtin: true, id: *h5t::$be }
            }
        }
    };
}

impl_hdf5type_for_builtin!(char, H5T_STD_U32LE, H5T_STD_U32BE);
impl_hdf5type_for_builtin!(bool, H5T_STD_U8LE, H5T_STD_U8BE);

impl_hdf5type_for_builtin!(i8, H5T_STD_I8LE, H5T_STD_I8BE);
impl_hdf5type_for_builtin!(i16, H5T_STD_I16LE, H5T_STD_I16BE);
impl_hdf5type_for_builtin!(i32, H5T_STD_I32LE, H5T_STD_I32BE);
impl_hdf5type_for_builtin!(i64, H5T_STD_I64LE, H5T_STD_I64BE);

impl_hdf5type_for_builtin!(u8, H5T_STD_U8LE, H5T_STD_U8BE);
impl_hdf5type_for_builtin!(u16, H5T_STD_U16LE, H5T_STD_U16BE);
impl_hdf5type_for_builtin!(u32, H5T_STD_U32LE, H5T_STD_U32BE);
impl_hdf5type_for_builtin!(u64, H5T_STD_U64LE, H5T_STD_U64BE);

#[cfg(target_pointer_width = "32")]
impl_hdf5type_for_builtin!(usize, H5T_STD_U32LE, H5T_STD_U32BE);
#[cfg(target_pointer_width = "64")]
impl_hdf5type_for_builtin!(usize, H5T_STD_U64LE, H5T_STD_U64BE);

impl_hdf5type_for_builtin!(f32, H5T_IEEE_F32LE, H5T_IEEE_F32BE);
impl_hdf5type_for_builtin!(f64, H5T_IEEE_F64LE, H5T_IEEE_F64BE);