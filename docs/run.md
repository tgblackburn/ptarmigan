# Running a Ptarmigan simulation

To launch a Ptarmigan simulation, switch to the directory and run:

```bash
cd ptarmigan
[mpirun -n np] ./target/release/ptarmigan path/to/input.yaml
```

This will be parallelized over `np` MPI tasks (if MPI support has been enabled).