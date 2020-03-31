# ptarmigan

Ponderomotive trajectories and radiation emission

## Build

The following need to be installed:

* an MPI library

Ptarmigan has been tested with OpenMPI and MPICH.

All other dependencies are Rust crates that are downloaded automatically by Cargo. Then building should be as simple as:

```bash
cargo build --release [-j NUM_THREADS]
```

where `NUM_THREADS` is the number of separate threads that Cargo is allowed to spawn.
