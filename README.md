# ptarmigan

Ponderomotive trajectories and radiation emission

## Build

All of ptarmigan's default dependencies are Rust crates, which are downloaded automatically by Cargo. Building the code in this case is as simple as running:

```bash
cargo build --release [-j NUM_THREADS]
```

where `NUM_THREADS` is the number of separate threads that Cargo is allowed to spawn.

There are two optional features, `with-mpi` and `fits-output`, which need external libraries to be installed. The former allows the code to run in parallel; the latter switches the default output format from text to FITS. Requirements are an MPI library (ptarmigan is tested against MPICH) and [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/). To build with one or both, run:

```bash
cargo build --release --features with-mpi,fits-output [-j NUM_THREADS]
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
