use hdf5_writer;
use hdf5_writer::{
    GroupHolder,
    ScatteredDataset
};

#[cfg(feature = "with-mpi")]
use mpi::traits::*;

#[cfg(not(feature = "with-mpi"))]
extern crate no_mpi as mpi;

#[cfg(not(feature = "with-mpi"))]
use mpi::Communicator;

fn main() {
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank();

    #[cfg(feature = "with-mpi")]
    println!("* Compiled with MPI support");
    #[cfg(not(feature = "with-mpi"))]
    println!("* Not compiled with MPI support");

    let data: Vec<f64> = if rank < 1 {
        vec![]
    } else {
        vec![rank as f64; 4]
    };

    let average = 42_f64;

    println!("{} got {:?}", rank, data);

    let file = hdf5_writer::ParallelFile::create(&world, "test.h5").unwrap();
    let group = file.new_group("folder").unwrap();

    let dataset = group.new_dataset("data").unwrap();
    dataset.write(&data[..]).unwrap();

    let scalar = group.new_dataset("average").unwrap();
    scalar.with_desc("not really").unwrap().only_task(0).write(&average).unwrap();

    let label = group.new_dataset("label").unwrap();
    label.only_task(0).write("test HDF5 file").unwrap();

    drop(group);
    drop(file);

    let file = hdf5_writer::ParallelFile::open(&world, "test.h5").unwrap();
    let group = file.open_group("folder").unwrap();

    let dataset = group.open_dataset("data").unwrap();
    if let Ok(ScatteredDataset { data, dims }) = dataset.read::<[f64]>() {
        println!("{} got {:?}, dims {:?}", rank, data, dims);
    };

    let dataset = group.open_dataset("average").unwrap();
    if let Ok(data) = dataset.read::<f64>() {
        println!("{} got {:?}", rank, data);
    }

    let attr = dataset.open_attribute("desc").unwrap();
    if let Ok(data) = attr.read::<String>() {
        println!("{} got {:?}", rank, data);
    }

    let dataset = group.open_dataset("label").unwrap();
    if let Ok(data) = dataset.read::<String>() {
        println!("{} got {:?}", rank, data);
    }

}