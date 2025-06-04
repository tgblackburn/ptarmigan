# Generating output

Ptarmigan has two main modes for generating output.
[Complete information](#Complete_information) about all particles in the final state (positions, momenta, etc) can be written to a single structured file.
The code can also generate 1D or 2D spectra from the final-state particle populations: these [distributions](#Distributions) are specified on a per-species basis and are written as individual files.
All output is written to the directory where the input file is found.

The following options apply to both kinds of output:

* `ident` (optional, default = no prefix): prepends a identifier string to the filenames of all produced output. Uses the name of the input file if `auto` is specified.
* `min_energy` (optional, default = `0.0`): if specified, discard secondary particles below a certain energy before creating the output distributions.
* `max_angle` (optional, default = `pi`): if specified, discard secondary particles that are moving, with respect to the shower's primary particle, at angles greater than the given limit.
* `coordinate_system` (optional, default = `laser`): by default, particle positions and momenta are output in the simulation coordinate system, where the laser travels towards positive z. If set to `beam`, these are transformed such that the beam propagation defines the positive z direction.
* `discard_background` (optional, default = `false`): whether to discard primary electrons that have not radiated, or primary photons that have not pair-created, before generating output.
`discard_background_e`, which applies to electrons only, is accepted for backwards compatibility but has lower priority than `discard_background`.
* `units` (optional, default = `auto`): select the units to be used when generating distribution or particle output. Possible choices of unit system are `hep` (distances in mm, momenta in GeV/c, etc), `si` (distances in m, momenta in kg/m/s, etc) or `auto` (distances in m, momenta in MeV/c, etc).
In future, it will be possible to select each unit individually.

### Complete information

These options control the output of complete information:

* `dump_all_particles` (optional): if present, information about all particles in the simulation will be written to file in the specified format. Possible formats are: `hdf5` (only available if Ptarmigan has been compiled with the feature `hdf5-output`). A brief guide to the structure and use of the HDF5 output file is provided [here](../output.md#hdf5).
* `dump_decayed_photons` (optional, default = `false`): if true, information about photons not in the final state (i.e. photons that have pair-created) will be included in the above output file.

### Distributions

These options control the output of particle spectra (distribution functions).

* `file_format`: select how to output particle distribution functions. Possible formats are: `plain_text` or `fits`.

The desired distribution outputs are specified per species:

* `electron` (optional): list of specifiers of the form `dstr1[:dstr2][:(log|auto;weight)]`, each of which should correspond to a distribution function. For example, `x:px` requests the distribution of the x coordinate and the corresponding momentum component. Each separate output is written to a separate file.
* `photon` (optional): as above.
* `positron` (optional): as above.
* `intermediate` (optional): as above. Provides information about photons that do not escape the laser pulse (i.e. that have pair-created).

The possible distributions `dstr` are:

* `x`, `y` and `z`: particle spatial coordinates, in metres
* `px`, `py`, `pz`: particle momenta, in MeV/c
* `energy`: particle energy, in MeV
* `gamma`: ratio of particle energy to electron mass, dimensionless
* `p^-` and `p^+`: particle lightfront momenta, in MeV/c
* `p_perp`: particle perpendicular momentum, i.e. `sqrt(px^2+py^2)`, in MeV/c
* `r_x`, `r_y`: ratio of perpendicular to lightfront momenta, `px / p^-` and `py / p^-`, dimensionless
* `r_perp`: `sqrt(r_x^2 + r_y^2)`, dimensionless
* `angle_x`, `angle_y`: angle between particle momentum and the z-axis, in radians
* `angle`: polar angle between particle momentum and the z-axis, in radians
* `pi_minus_angle` (`theta` also accepted): polar angle between particle momentum and the *negative* z-axis, in radians
* `birth_a`: normalized amplitude a<sub>0</sub> at the point where the particle was created:
either the cycle-averaged (RMS) value (if using LMA) or the instantaneous value, `e E / m c omega` (if using LCFA).
* `parent_chi`: the quantum parameter of this particle's parent, at the point where this particle was created: eithe the cycle-averaged value (if using the LMA) or the instantaneous value (if using the LCFA).
* `S_1`, `S_2` and `S_3`: the Stokes parameters associated with the particle polarization. `S_1` is associated with linear polarization along x (+1) or y (-1); `S_2` with linear polarization at 45 degrees to these axes; and `S_3` to the degree of circular polarization.
In the current version of Ptarmigan, these are meaningful only for photons.
* `absorption`: the amount of energy the particle has absorbed from the laser pulse.

It is possible to generate weighted distributions, e.g. `x:y:(energy)`, by passing an additional, bracketed, argument to the output specifier.
The possible weight functions are:

* `auto`: the particle weight (default)
* `energy`: particle energy, in MeV
* `pol_x`: the projection of the particle polarization along the global x-axis
* `pol_y`: the projection of the particle polarization along the global y-axis
* `helicity`: the projection of the particle polarization along its momentum

The number of bins, or whether they should be log-scaled, is controlled by adding an integer or `log` *before* the weight specification.
The weight function must be given explicitly in this case, e.g. `energy:(log;auto)`.

A simple range cut can be applied before binning by adding a third argument inside the brackets, e.g. `energy:(auto; auto; angle in 0, max)`.
Both the number of bins and the weight must be given (though they can be replaced with `auto`).
The syntax is `var in min, max`, where `var` is one of the particle properties given above and `min`, `max` are math expressions evaluated at run time.
Only particles that are within the given bounds are binned.
All particles below or above can be accepted by replacing the relevant bound with `auto`.

## Summary statistics

The `stats` block works in a similar way to the `output` block.
If it contains a list of specifiers, given on a per-species basis,

* `electron` (optional): list of specifiers
* `photon` (optional): list of specifiers
* `positron` (optional): list of specifiers
* `intermediate` (optional): list of specifiers

Ptarmigan will write aggregated statistical information about the particle final state distributions to a file called 'stats.txt' (with the appropriate identifier prefix, if specified in [output](#output)) in the same directory as the input file.

Each specifier must be one of:

* ``op var[`weight]``
* ``op var[`weight] in (min; max)``
* ``op var[`weight] for var2 in (min; max)``

where `op` is one of:

* `total`
* `fraction`
* `mean`
* `variance`
* `minimum`, `maximum`
* `circmean`, `circvariance`, `circstd`

and `var` is a desired output (`px`, the x-component of momentum, for example).
The range of values to be used can be specified by `var`, or another output entirely, `var2`.
Both `min` and `max` can be arbitrary mathematical expressions, using values given in the [constants](#constants) block, or `auto`, in which case the range is detected automatically.
They are assumed to be given in SI units: use conversion constants, e.g. `MeV` or `MeV/c` for energies and momenta, if necessary.
The contribution of each particle to the statistic is either its weight (i.e. number) or may given in terms of another variable.

For example: `mean energy` computes the average of the particle energy; ``variance angle_x`energy`` computes the energy-weighted variance of the angle between the particle momentum and the x axis; `mean px in (1.0 * MeV/c; auto)` computes the average px for all particles that have px greater than 1 MeV/c; `total number for px in (1.0 * MeV/c; 2.0 * MeV/c)` calculates the number of particles with momentum component between the specified bounds.

The summary statistics `circmean`, `circvariance` and `circstd`, which compute the circular mean, variance and standard deviation respectively, are meaningful only for angular quantities.
The circular variance is defined as $ 1- \langle R \rangle$, and the circular standard deviation as $\sqrt{-2\ln\langle R \rangle}$, where $\langle R \rangle$ is the mean resultant vector.
This follows the definition from [Statistics of Directional Data; K.V. Mardia](https://doi.org/10.1016/C2013-0-07425-7).

Expressions involving defined constants from the [constants](#constants) block can also be calculated and written to the 'stats.txt' file.  The identifier prefix is

* `expression` (optional): list of specifiers

The specifier must be one of:

* ``name[`formula] expression``
* ``name[`formula] expression unit``

where `name` is a name for the expression being evaluated, `expression` is the combination of constants to be evaluated and `unit` is the unit of the expression result (it is up to the user to ensure this is correctly specified). There should be no whitespace in the expression. `` `formula`` is an optional tag to write the expression formula as well as the calculated value.

For example, `init_energy initial_gamma*me*c^2/MeV MeV` would compute and write the value of this expression, in units of MeV, with name `init_energy` to the 'stats.txt' file, provided `initial_gamma` was specified in the [constants](#constants) block. The unit would also be printed as MeV. Adding the formula tag, ``init_energy`formula initial_gamma*me*c^2/MeV MeV``, will also write `initial_gamma*me*c^2/MeV` as a string to the 'stats.txt' file.
