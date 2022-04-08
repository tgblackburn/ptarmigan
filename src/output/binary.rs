//! Output of binary-formatted particle data

/// Represents a file handle, to which raw numerical data can be written.
pub trait OutputHandle<'a> {
    /// Opens a new dataset in the associated file handle.
    fn new_data(&'a self, name: &str) -> DatasetHandle<'a, Self> {
        DatasetHandle {
            output: self,
            name: name.to_owned(),
            unit: None,
            desc: None,
            condition: true
        }
    }
}

impl OutputHandle<'_> for hdf5::Group {}

/// Represents an open dataset
pub struct DatasetHandle<'a, H> where H: ?Sized {
    output: &'a H,
    name: String,
    unit: Option<String>,
    desc: Option<String>,
    condition: bool,
}

impl<'a> DatasetHandle<'a, hdf5::Group> {
    /// Assign a unit to the data output
    pub fn with_unit(mut self, unit: &str) -> Self {
        self.unit = Some(unit.to_owned());
        self
    }

    /// Assign a description of the dataset
    pub fn with_desc(mut self, desc: &str) -> Self {
        self.desc = Some(desc.to_owned());
        self
    }

    /// Means that the dataset will only be written if the closure returns True
    pub fn with_condition<F>(mut self, f: F) -> Self where F: FnOnce() -> bool {
        self.condition = f();
        self
    }

    /// Writes data (a scalar value `&T`, slice `&[T]` or string slice `&str`) to the current
    /// dataset handle, returning the output handle so that further datasets can be written.
    pub fn write<T>(self, data: &T) -> Result<&'a hdf5::Group, hdf5::Error>
    where T: AsHdf5Data<'a> + ?Sized
    {
        if self.condition {
            data.write_into(self.output, self.name.as_ref(), self.unit.as_deref(), self.desc.as_deref())
        } else {
            Ok(self.output)
        }
    }
}

pub trait AsHdf5Data<'a> {
    fn write_into(&self, group: &'a hdf5::Group, name: &str, unit: Option<&str>, desc: Option<&str>) -> hdf5::Result<&'a hdf5::Group>;
}

impl<'a, T> AsHdf5Data<'a> for T where T: hdf5::types::H5Type {
    fn write_into(&self, group: &'a hdf5::Group, name: &str, unit: Option<&str>, desc: Option<&str>) -> hdf5::Result<&'a hdf5::Group> {
        use std::str::FromStr;
        use hdf5::types::VarLenUnicode;

        let dataset = group.new_dataset::<T>()
            .create(name)?;

        let unit = unit.map(|s| VarLenUnicode::from_str(s).ok()).flatten();
        let desc = desc.map(|s| VarLenUnicode::from_str(s).ok()).flatten();
        
        if let Some(s) = unit {
            dataset.new_attr::<VarLenUnicode>()
                .create("unit")?
                .write_scalar(&s)?;
        }

        if let Some(s) = desc {
            dataset.new_attr::<VarLenUnicode>()
                .create("desc")?
                .write_scalar(&s)?;
        }
        
        dataset.write_scalar(self)?;
        Ok(group)
    }
}

impl<'a> AsHdf5Data<'a> for str {
    fn write_into(&self, group: &'a hdf5::Group, name: &str, unit: Option<&str>, desc: Option<&str>) -> hdf5::Result<&'a hdf5::Group> {
        use std::str::FromStr;
        use hdf5::types::VarLenUnicode;

        let str = VarLenUnicode::from_str(self).ok();
        let unit = unit.map(|s| VarLenUnicode::from_str(s).ok()).flatten();
        let desc = desc.map(|s| VarLenUnicode::from_str(s).ok()).flatten();

        let dataset = group.new_dataset::<VarLenUnicode>()
            .create(name)?;

        if let Some(s) = str {
            dataset.write_scalar(&s)?;
        }

        if let Some(s) = unit {
            dataset.new_attr::<VarLenUnicode>()
                .create("unit")?
                .write_scalar(&s)?;
        }

        if let Some(s) = desc {
            dataset.new_attr::<VarLenUnicode>()
                .create("desc")?
                .write_scalar(&s)?;
        }

        Ok(group)
    }
}

impl<'a, T> AsHdf5Data<'a> for [T] where T: hdf5::types::H5Type {
    fn write_into(&self, group: &'a hdf5::Group, name: &str, unit: Option<&str>, desc: Option<&str>) -> hdf5::Result<&'a hdf5::Group> {
        use std::str::FromStr;
        use hdf5::types::VarLenUnicode;

        let dataset = group.new_dataset_builder()
            .with_data(self)
            .create(name)?;

        let unit = unit.map(|s| VarLenUnicode::from_str(s).ok()).flatten();
        let desc = desc.map(|s| VarLenUnicode::from_str(s).ok()).flatten();
        
        if let Some(s) = unit {
            dataset.new_attr::<VarLenUnicode>()
                .create("unit")?
                .write_scalar(&s)?;
        }

        if let Some(s) = desc {
            dataset.new_attr::<VarLenUnicode>()
                .create("desc")?
                .write_scalar(&s)?;
        }

        Ok(group)
    }
}