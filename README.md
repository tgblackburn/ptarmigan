# ptarmigan

Ponderomotive trajectories and radiation emission

## Build

All of ptarmigan's default dependencies are Rust crates, which are downloaded automatically by Cargo. Building the code in this case is as simple as running:

```bash
cargo build --release [-j NUM_THREADS]
```

where `NUM_THREADS` is the number of separate threads that Cargo is allowed to spawn.

There are three optional features, `with-mpi`, `fits-output` and `hdf5-output`, which need external libraries to be installed. The first allows the code to run in parallel; the second switches the default output format for distribution functions from text to FITS. Requirements are an MPI library (ptarmigan is tested against MPICH), [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/) and [libhdf5](https://www.hdfgroup.org/solutions/hdf5/). To build with one or all of these, run:

```bash
cargo build --release --features with-mpi,fits-output,hdf5-output [-j NUM_THREADS]
```

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
