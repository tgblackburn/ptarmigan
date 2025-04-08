# Importing a particle beam

In this case, it is necessary to specify:

* `species` (primary particle type, must be one of `electron`, `photon` or `positron`)

and optionally

* `collision_angle`: see [here](beam_generation.md#collision-parameters) for details
* `offset`: see [here](beam_generation.md#collision-parameters) for details

in the `beam` section of the input file.
All other quantities will be imported from the external file.

At present, Ptarmigan will accept only HDF5-formatted input.
The HDF5 file in question must either be the output of a Ptarmigan simulation, or have compatible structure.

?> [PICA](https://github.com/hixps/pica) (Polarized ICS CAlculator) produces output that is compatible with Ptarmigan.

In a sub-section of `beam` named `from_hdf5`, provide:

* `from_hdf5`:
  * `file`: path to the HDF5 file that stores the particle data, relative to the location of the input file.
  * `distance_between_ips`: the distance between the origin of the coordinate system, used by the imported particle beam, and the laser collision point, in metres.
  This is used to propagate the particles between the interaction points, assuming ballistic drift.
  A `distance_between_ips` of `0.0` is perfectly fine: it means that the particle positions are defined with respect to the laser collision point.
  * `auto_timing` (optional, default = `true`): disable this to prevent Ptarmigan propagating the particles between the interaction points.
  Positions specified in the external file will be respected: for example, if a particle has a position `[0.0, 0.0, 0.0, 0.0]`, it will be initialised inside the laser pulse at time zero.
  * `min_energy` (optional, default = `0.0`): if specified, skip any particles that have less energy than this threshold, during import.
  * `max_angle` (optional, default = `pi`): if specified, skip particles that are moving, with respect to the particle beam axis, at angles greater than the given limit.

## Creating a suitable input file

So, you have a particle beam that cannot be represented by Ptarmigan's standard input parameters?
Perhaps the six-dimensional phase space is highly correlated, or you have output from another code you want to use?

Creating an HDF5 file of particle data that contains the following datasets:

* `beam_axis`: a string that is one of `+x`, `-x`, `+z` or `-z`, indicating the beam propagation direction.
* `config/unit/momentum`: a UTF-8 formatted string that gives the units of the four-momentum.
Ptarmigan will recognise `kg/m/s`, as well as `MeV/c` and `GeV/c` (with or without the `/c`).
* `config/unit/position`: a UTF-8 formatted string that gives the units of the four-position.
Ptarmigan will recognise `um`, `micron`, `mm` and `m`.
* `final-state/{particle}/weight`: an N⁢ × 1 array of doubles that gives the particle weights, i.e. the number of real particles represented.
* `final-state/{particle}/momentum`: an N⁢ × 4 array of doubles that gives the particle four-momenta $(\gamma m c, \gamma m \vec{v})$ or $(E/c, \vec{p})$.
* `final-state/{particle}/position`: an N⁢ × 4 array of doubles that gives the particle four-positions $(c t, \vec{r})$.

and optionally:

* `final-state/{particle}/polarization`: an N⁢ × 4 array of doubles that gives the particle Stokes parameters $(1, S_1, S_2, S_3)$.
$S_1$ and $S_2$ represent the degree of linear polarization and $S_3$ the degree of circular polarization.

will allow it to be imported by Ptarmigan.

`{particle}` should be one of `electron`, `positron` or `photon`.
The species that is to be imported is specified in the Ptarmigan input file.