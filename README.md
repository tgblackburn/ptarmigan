# Ptarmigan

![OpenMPI build status](https://github.com/tgblackburn/ptarmigan/actions/workflows/open-mpi.yml/badge.svg) ![MPICH build status](https://github.com/tgblackburn/ptarmigan/actions/workflows/mpich.yml/badge.svg) ![version](https://img.shields.io/github/v/release/tgblackburn/ptarmigan?include_prereleases) ![license](https://img.shields.io/github/license/tgblackburn/ptarmigan)

Simulate the interaction between a high-energy particle beam and an intense laser pulse, including the classical dynamics and strong-field QED processes.

<p align="center">
  <img src="docs/img/collision.png" alt="A laser pulse (left) collides with a beam of electrons (right), lauching an electromagnetic shower">
</p>

## What's included

A summary of Ptarmigan's physics coverage can be found [here](docs/physics.md).

## Build

Ptarmigan's default dependencies are Rust crates, so the code can be built simply by running the following command:

```bash
cargo build --release [--features with-mpi,hdf5-output]
```

There are two optional features, which require additional dependencies to be installed:

* `with-mpi`, which enables parallel processing via MPI. 
* `hdf5-output`, which enables output of complete particle data as an HDF5 file.

Complete build instructions can be found [here](docs/build.md).
The Ptarmigan changelog can be found [here](docs/changelog.md).

## Specify problem

Ptarmigan takes as its single argument the path to a YAML file describing the input configuration.
Output is automatically written to the same directory as this file.
The structure of the input file is described in detail [here](docs/input_guide/README.md).
The inputs for some test problems can be found in [examples](examples).

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

## Reference

The main reference for Ptarmigan is

> T. G. Blackburn, B. King and S. Tang,
"Simulations of laser-driven strong-field QED with Ptarmigan: Resolving wavelength-scale interference and ɣ-ray polarization,"
[Physics of Plasmas 30, 093903 (2023)](https://doi.org/10.1063/5.0159963),
[arXiv:2305.13061 \[hep-ph\]](https://arxiv.org/abs/2305.13061)

and individual releases are archived on [Zenodo](https://doi.org/10.5281/zenodo.7956999).