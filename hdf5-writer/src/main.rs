use hdf5_writer;
use hdf5_writer::GroupHolder;

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
        vec![5.0, 6.0, 7.0, 8.0]
    };

    println!("{} got {:?}", rank, data);

    let file = hdf5_writer::ParallelFile::create(&world, "test.h5").unwrap();
    let group = file.new_group("folder").unwrap();
    let dataset = group.new_dataset("data").unwrap();
    dataset.write(&data[..]).unwrap();
}