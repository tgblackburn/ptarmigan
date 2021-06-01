# ptarmigan

Ponderomotive trajectories and radiation emission

## Build

All of ptarmigan's default dependencies are Rust crates, which are downloaded automatically by Cargo. Building the code in this case is as simple as running:

```bash
cargo build --release [-j NUM_THREADS]
```

where `NUM_THREADS` is the number of separate threads that Cargo is allowed to spawn.

There are three optional features: `with-mpi`, which enables parallel processing via MPI; `fits-output`, which switches the output format for distribution functions from text to FITS; and `hdf5-output`, which switches the output format for particle arrays to HDF5.
Each requires some external libraries to be installed.
`with-mpi` needs an MPI library (ptarmigan is tested against OpenMPI, versions <= 3.1, and MPICH) and the Clang compiler.
`fits-output` and `hdf5-output` require [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/) and [libhdf5](https://www.hdfgroup.org/solutions/hdf5/) respectively.
To build with one or all of these, run:

```bash
cargo build --release --features with-mpi,fits-output,hdf5-output [-j NUM_THREADS]
```

The ptarmigan changelog can be found [here](docs/changelog.md).

## Specify problem

ptarmigan takes as its single argument the path to a YAML file describing the input configuration. Output is automatically written to the same directory as this file. The inputs for some test problems can be found in [examples](examples). Starting from scratch, the input needs to contain the following sections:

* control
* laser
* beam

and optionally

* constants

The structure of the input file is described in detail [here](docs/input.md).

## Run

Assuming ptarmigan has been downloaded to `ptarmigan` and already built,

```bash
cd ptarmigan
[mpirun -n np] ./target/release/ptarmigan path/to/input.yaml
```

will run the code, parallelized over `np` MPI tasks (if MPI support has been enabled).
