# Building Ptarmigan

All of Ptarmigan's default dependencies are Rust crates, which are downloaded automatically by Cargo. Building the code in this case is as simple as running:

```bash
cargo build --release [-j NUM_THREADS]
```

where `NUM_THREADS` is the number of separate threads that Cargo is allowed to spawn.

## Optional features

The following optional features are available:

* `with-mpi`, which enables parallel processing via MPI. Requires an MPI library (Ptarmigan is tested against OpenMPI and MPICH) and the Clang compiler.
* `hdf5-output`, which enables output of complete particle data as an HDF5 file. Requires [libhdf5](https://www.hdfgroup.org/solutions/hdf5/).
If `with-mpi` and `hdf5-output` are both specified, the HDF5 library must have been compiled with MPI support.

To build with a combination of these features, run:

```bash
cargo build --release --features with-mpi,hdf5-output [-j NUM_THREADS]
```

## HPC systems

Coming soon!

## Windows

Start by installing the following components from the [Visual Studio Build Tools 2022](https://visualstudio.microsoft.com/visual-cpp-build-tools/):

* MSVC v143 - VS 2022 C++ x64/x86 build tools
* Windows 11 SDK

Then get [Rust](https://www.rust-lang.org/tools/install) itself.
This will suffice to build Ptarmigan without optional features (`cargo build --release`).

In order to run Ptarmigan in parallel, you will need to install, in addition to the above:

* [Microsoft MPI v10.1.2](https://www.microsoft.com/en-us/download/details.aspx?id=100593) (both the SDK and the redistributable)
* [LLVM v14.0.6](https://github.com/llvm/llvm-project/releases/tag/llvmorg-14.0.6)

Ptarmigan can then be built with MPI support (`cargo build --release --features with-mpi`).