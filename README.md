# Ptarmigan

![OpenMPI build status](https://github.com/tgblackburn/ptarmigan/actions/workflows/open-mpi.yml/badge.svg) ![MPICH build status](https://github.com/tgblackburn/ptarmigan/actions/workflows/mpich.yml/badge.svg) ![version](https://img.shields.io/github/v/release/tgblackburn/ptarmigan?include_prereleases) ![license](https://img.shields.io/github/license/tgblackburn/ptarmigan)

Simulate the interaction between a high-energy particle beam and an intense laser pulse, including the classical dynamics and strong-field QED processes.

<p align="center">
  <img src="docs/collision.png" alt="A laser pulse (left) collides with a beam of electrons (right), lauching an electromagnetic shower">
</p>

## Build

All of Ptarmigan's default dependencies are Rust crates, which are downloaded automatically by Cargo. Building the code in this case is as simple as running:

```bash
cargo build --release [-j NUM_THREADS]
```

where `NUM_THREADS` is the number of separate threads that Cargo is allowed to spawn.

The following optional features are available:

* `with-mpi`, which enables parallel processing via MPI. Requires an MPI library (Ptarmigan is tested against OpenMPI and MPICH) and the Clang compiler.
* `hdf5-output`, which enables output of complete particle data as an HDF5 file. Requires [libhdf5](https://www.hdfgroup.org/solutions/hdf5/).
If `with-mpi` and `hdf5-output` are both specified, the HDF5 library must have been compiled with MPI support.

To build with a combination of these features, run:

```bash
cargo build --release --features with-mpi,hdf5-output [-j NUM_THREADS]
```

The Ptarmigan changelog can be found [here](docs/changelog.md).

Instructions for building the code on Windows can be found [here](docs/win10_build_guide.md).

## Specify problem

Ptarmigan takes as its single argument the path to a YAML file describing the input configuration. Output is automatically written to the same directory as this file. The inputs for some test problems can be found in [examples](examples). Starting from scratch, the input needs to contain the following sections:

* laser
* beam

and optionally

* control
* constants
* output
* stats

The structure of the input file is described in detail [here](docs/input.md).

## Run

Assuming Ptarmigan has been downloaded to `ptarmigan` and already built,

```bash
cd ptarmigan
[mpirun -n np] ./target/release/ptarmigan path/to/input.yaml
```

will run the code, parallelized over `np` MPI tasks (if MPI support has been enabled).

## Output

The code bins the final-state particles to generate the distribution functions requested in the input file, which are written in plain-text or FITS format.

If `hdf5-output` is enabled, complete data about all particles can be written as a single HDF5 file.


## Contribute

Pull requests, bug fixes and new features, are welcome!

Contributors:

* Tom Blackburn
* Kyle Fleck
