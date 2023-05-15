# Creating an input configuration

Ptarmigan takes as its single argument the path to a YAML file describing the input configuration. This file is divided into sections that set up the initial particle distributions, externally injected electromagnetic fields, and desired physical processes. [laser](#laser), [beam](#beam) and [output](#output) are compulsory.

## control

Physics:

* `radiation_reaction` (optional, default = `true`): set to `false` to disable electron/positron recoil on photon emission.
* `pair_creation` (optional, default = `true`): set to `false` to disable photon tracking and electron-positron pair creation.
* `pol_resolved` (optional, default = `false`): enable the use of photon-polarization-resolved rates, which improves simulation accuracy by approximately 20%.
* `classical` (optional, default = `false`): use the classical photon emission rate. If `radiation_reaction` is enabled, electrons and positrons lose energy smoothly, following the Landau-Lifshitz equation. Disables pair creation unless otherwise specified.
A modified classical model can be chosen by setting `classical` to `gaunt_factor_corrected`.
In this model the instantaneous radiated power is reduced by the Gaunt factor g(χ) and the upper bound on the photon spectrum corrected to the electron energy.
This option is only available under the LCFA.
* `lcfa` (optional, default = `false`): if `true`, use rates calculated in the locally constant, crossed fields approximation to model QED processes.
* `bandwidth_correction` (optional, default = `false`, ignored if `lcfa: true`): if `true`, correct the photon momentum sampling algorithm to account for the laser pulse's finite bandwidth. Has no effect if LCFA rates are selected.

Numerics:

* `dt_multiplier` (optional, default = `1.0`): the size of the timestep is set automatically by the code to ensure accuracy of the particle pusher; this applies a scaling factor to it.
* `increase_pair_rate_by` (optional, default = `1.0`): if specified, increases the pair creation rate, while decreasing the weight of any created electrons and positrons, by the same factor. This helps resolve the positron spectrum when the total probability is much smaller than 1/N, where N is the number of primary particles. A setting of `auto` will be replaced by a suitable default value, as determined from the laser amplitude and particle energy. In principle, an arbitrarily large increase may be specified, because the code automatically adjusts it if the probability per timestep becomes too large. However, this will mean that a very large number of (low-weight) electrons and positrons will be generated and tracked.
* `rng_seed` (optional, default = `0`): an unsigned integer that, if specified, is used as the basis for seeding the PRNG.

Tracking:

* `select_multiplicity` (optional): to facilitate comparisons with theory, select only those showers with the desired number of daughter particles when creating output.
* `stop_at_time` (optional): if specified, stops tracking at the given instant of time. Otherwise, the simulation tracks particles until they have travelled through the entire laser pulse. Time zero corresponds to the point at which the peak of the laser passes through the focal plane.

## laser

* `a0`: the laser strength parameter, normalized amplitude, etc.
Defined by the peak electric field, so a0 is sqrt(2) larger for LP than for CP at fixed intensity.
Alternatively, specifying `a0:start`, `a0:step` and `a0:stop` will run a single simulation for all a0 values in the given range.
* `wavelength`: of the carrier wave, in metres, or
* `omega`: the equivalent photon energy, in joules. The conversion constants `eV` etc are provided for convenience.
* `waist` (optional): if specified, the laser pulse will be focused to a spot size of `waist`, which defines the radius at which the intensity falls to 1/e^2 of its maximum value. Otherwise the laser is modelled as a plane wave.
* `envelope` (optional, default is `cos^2` in 1D and `gaussian` in 3D): the temporal envelope of the laser pulse. Select one of:
    * `cos^2`
    * `flattop` (constant intensity over the specified number of cycles, with a one-wavelength long, smooth ramp-up and ramp-down)
    * `gaussian`
* `fwhm_duration` (if `envelope: gaussian`): the full width at half max of the *intensity envelope*, in seconds.
* `n_cycles` (if `envelope: cos^2` or `flattop`): the total duration of the pulse, expressed in wavelengths. Usually (but not required to be) an integer.
* `chirp_coeff` (optional, ignored if `waist` is specified): specifies `b`, the chirp coefficient, which appears in the total phase `ϕ + b ϕ^2` of the laser carrier wave. A positive `b` leads to an instantaneous frequency that increases linearly from head to tail.
* `polarization`: the polarization of the carrier wave, either `linear` (along `x`) or `circular`.

## beam

* `n`: number of primary particles. `ne` is also accepted.
* `species` (optional, default = `electron`): primary particle type, must be one of `electron`, `photon` or `positron`.
* `charge` (optional): if specified, weight each primary electron such that the whole ensemble represents a bunch of given charge. (Include a factor of the elementary charge `e` to get a specific number.)
* `gamma`: the mean Lorentz factor.
* `sigma` (optional, default = `0.0`): the standard deviation of the electron Lorentz factors, set to zero if not specified.
* `bremsstrahlung_source` (optional, if primary particles are photons, default = `false`): switches energy spectrum from Gaussian to mimic a bremsstrahlung source.
* `gamma_min` (required if `bremsstrahlung_source` is `true`): lower bound for the bremsstrahlung energy spectrum.
* `radius` (optional, default = `0.0`): if a single value is specified, the beam is given a cylindrically symmetric Gaussian charge distribution, with specified standard deviation in radius (metres). The distribution is set explicitly if a tuple of `[radius, dstr]` is given. `dstr` may be either `normally_distributed` (the default) or `uniformly_distributed`. In the latter case, `radius` specifies the maximum, rather than the standard deviation.
The distribution (if normal) may be optionally truncated by specifying `[radius, normally_distributed, max_radius]`.
* `length` (optional, default = `0.0`): standard deviation of the (Gaussian) charge distribution along the beam propagation axis (metres)
* `energy_chirp` (optional, default = `0.0`): if specified, introduces a correlation of the requested magnitude between the particle's energy and its longitudinal offset from the beam centroid. A positive chirp means that the head of the beam (which hits the laser first) has higher energy than the tail. The specified value must be between -1 and +1.
* `collision_angle` (optional, default = `0.0`): angle between beam momentum and laser axis in radians, with zero being perfectly counterpropagating; the constant `degree` is provided for convenience.
* `rms_divergence` (optional, default = `0.0`): if specified, the angles between particle initial momenta and the beam propagation axis are normally distributed, with given standard deviation.
* `offset` (optional, default = `[0.0, 0.0, 0.0]`): introduces an alignment error between the particle beam and the laser pulse, as defined by the location of the beam centroid at the time when the peak of the laser pulse passes through focus.
The offsets are defined with respect to the beam propagation axis: the first two components are perpendicular to this axis and the third is parallel to it.
For example, if the offset is `[0.0, 0.0, delta > 0]` and the collision angle is `0.0`, the peak of the laser reaches the focal plane before the beam centroid does; the collision, while perfectly aligned in the perpendicular directions, is delayed by time `delta/(2c)`.
* `stokes_pars` (optional, default = `[0.0, 0.0, 0.0]`): specifies the primary particles' polarization in terms of the three Stokes parameters `S_1`, `S_2` and `S_3` (equiv. `Q`, `U` and `V`).
The basis is defined with respect to the `x`-`z` plane and the particle velocity:
`S_1` is associated with linear polarization along x (`+1.0`) or y (`-1.0`); `S_2` with linear polarization at 45 degrees to these axes; and `S_3` to the degree of circular polarization.
For example, `[1.0, 0.0, 0.0]` loads particles that are polarized parallel to the laser electric field (if `laser:polarization` is `linear`).
The default behaviour is to assume that the particles are unpolarized.

## output

All output is written to the directory where the input file is found.

* `ident` (optional, default = no prefix): prepends a identifier string to the filenames of all produced output. Uses the name of the input file if `auto` is specified.
* `file_format`: select how to output particle distribution functions. Possible formats are: `plain_text` or `fits`.
* `min_energy` (optional, default = `0.0`): if specified, discard secondary particles below a certain energy before creating the output distributions.
* `max_angle` (optional, default = `pi`): if specified, discard secondary particles that are moving, with respect to the shower's primary particle, at angles greater than the given limit.
* `dump_all_particles` (optional): if present, information about all particles in the simulation will be written to file in the specified format. Possible formats are: `hdf5` (only available if Ptarmigan has been compiled with the feature `hdf5-output`). A brief guide to the structure and use of the HDF5 output file is explained in [this notebook](hdf5_import_guide.ipynb).
* `coordinate_system` (optional, default = `laser`): by default, particle positions and momenta are output in the simulation coordinate system, where the laser travels towards positive z. If set to `beam`, these are transformed such that the beam propagation defines the positive z direction.
* `discard_background` (optional, default = `false`): whether to discard primary electrons that have not radiated, or primary photons that have not pair-created, before generating output.
`discard_background_e`, which applies to electrons only, is accepted for backwards compatibility but has lower priority than `discard_background`.
* `units` (optional, default = `auto`): select the units to be used when generating distribution or particle output (FITS/HDF5-formatted). Possible choices of unit system are `hep` (distances in mm, momenta in GeV/c, etc), `si` (distances in m, momenta in kg/m/s, etc) or `auto` (distances in m, momenta in MeV/c, etc).
In future, it will be possible to select each unit individually.

The desired distribution outputs are specified per species:

* `electron` (optional): list of specifiers of the form `dstr1[:dstr2][:(log|auto;weight)]`, each of which should correspond to a distribution function. For example, `x:px` requests the distribution of the x coordinate and the corresponding momentum component. Each separate output is written to a separate file.
* `photon` (optional): as above.
* `positron` (optional): as above.

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
* `S_1`, `S_2` and `S_3`: the Stokes parameters associated with the particle polarization. `S_1` is associated with linear polarization along x (+1) or y (-1); `S_2` with linear polarization at 45 degrees to these axes; and `S_3` to the degree of circular polarization.
In the current version of Ptarmigan, these are meaningful only for photons.

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

## stats

If specified, writes aggregated statistical information about the particle final state distributions to a file called 'stats.txt' (with the appropriate identifier prefix, if specified in [output](#output)) in the same directory as the input file.

* `electron` (optional): list of specifiers
* `photon` (optional): list of specifiers
* `positron` (optional): list of specifiers

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

and `var` is a desired output (`px`, the x-component of momentum, for example). The range of values to be used can be specified by `var`, or another output entirely, `var2`. Both `min` and `max` can be arbitrary mathematical expressions, using values given in the [constants](#constants) block, or `auto`, in which case the range is detected automatically. The contribution of each particle to the statistic is either its weight (i.e. number) or may given in terms of another variable.

For example: `mean energy` computes the average of the particle energy; ``variance angle_x`energy`` computes the energy-weighted variance of the angle between the particle momentum and the x axis; `mean px in (1.0; auto)` computes the average px for all particles that have px greater than 1; `total number for px in (1.0; 2.0)` calculates the number of particles with momentum component between the specified bounds.

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

## constants

Everywhere an integer or floating-point number is requested in the input file, a named value may be given instead, provided that its value is specified in this section.

For example, `gamma: gamma0` in the [beam](#beam) section would be accepted provided that `gamma0` was appropriately defined.

Entries in the constants block can be plain numbers, mathematical expressions, or functions of previously defined entries. For example,
```yaml
constants:
  a0: 15.0
  gamma0: 1000.0 / 0.511
  max_angle: atan(a0 / gamma0)
```
would make the numerical values `a0 = 15.0`, `gamma = 1956.95` and `max_angle = 7.6649e-3` available throughout the input file.
The block is parsed once, in order, so variables cannot be forward-defined.
YAML syntax requires that keys are unique: in the case of a repeated variable, only the last definition will apply.

## Maths parser

The code makes use of [evalexpr](https://crates.io/crates/evalexpr) when parsing the input file. In addition to the functions and constants this crate provides, Ptarmigan provides:

* the physical constants `me`, `mp`, `c`, `e`: the electron mass, proton mass, speed of light and elementary charge, respectively, all in SI units.
* the conversion constants `eV`, `keV`, `MeV`, `GeV`, `femto`, `pico`, `nano`, `milli`.
