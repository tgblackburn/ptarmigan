//! Output of binary-formatted particle data

/// Represents a file handle, to which raw numerical data can be written.
pub trait OutputHandle<'a, T: ?Sized, S> {
    type Error;
    /// Writes data (a scalar value `&T`, slice `&[T]` or string slice `&str`) to specified output handle.
    fn write(&'a self, name: S, data: &T) -> Result<&'a Self, Self::Error>;
    /// Writes data (a scalar value `&T`, slice `&[T]` or string slice `&str`) to specified output handle,
    /// if `test` is `true`.
    fn write_if(&'a self, test: bool, name: S, data: &T) -> Result<&'a Self, Self::Error>;
}

impl<'a, T, S> OutputHandle<'a, T, S> for hdf5::Group
where
    T: AsHdf5Data<'a> + ?Sized,
    S: AsRef<str>
{
    type Error = hdf5::Error;

    fn write(&'a self, name: S, data: &T) -> Result<&'a Self, Self::Error> {
        data.write_into(self, name.as_ref())
    }

    fn write_if(&'a self, test: bool, name: S, data: &T) -> Result<&'a Self, Self::Error> {
        if test {
            data.write_into(self, name.as_ref())
        } else {
            Ok(self)
        }
    }
}

pub trait AsHdf5Data<'a> {
    fn write_into(&self, group: &'a hdf5::Group, name: &str) -> hdf5::Result<&'a hdf5::Group>;
}

impl<'a, T> AsHdf5Data<'a> for T where T: hdf5::types::H5Type {
    fn write_into(&self, group: &'a hdf5::Group, name: &str) -> hdf5::Result<&'a hdf5::Group> {
        group.new_dataset::<T>()
            .create(name)?
            .write_scalar(self)
            .map(|_| group)
    }
}

impl<'a> AsHdf5Data<'a> for str {
    fn write_into(&self, group: &'a hdf5::Group, name: &str) -> hdf5::Result<&'a hdf5::Group> {
        use std::str::FromStr;
        use hdf5::types::VarLenUnicode;
        match VarLenUnicode::from_str(self) {
            Ok(vlu) => {
                group.new_dataset::<VarLenUnicode>()
                    .create(name)?
                    .write_scalar(&vlu)
                    .map(|_| group)
            },
            Err(e) => Err(hdf5::Error::Internal(e.to_string()))
        }
    }
}

impl<'a, T> AsHdf5Data<'a> for [T] where T: hdf5::types::H5Type {
    fn write_into(&self, group: &'a hdf5::Group, name: &str) -> hdf5::Result<&'a hdf5::Group> {
        group.new_dataset::<T>()
            .create(name)?
            .write(self)
            .map(|_| group)
    }
}