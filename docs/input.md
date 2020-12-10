# Creating an input configuration

ptarmigan takes as its single argument the path to a YAML file describing the input configuration. This file is divided into sections that set up the initial particle distributions, externally injected electromagnetic fields, and desired physical processes. [laser](#laser), [beam](#beam) and [output](#output) are compulsory.

## control

* `dt_multiplier` (optional, default = `1.0`): the size of the timestep is set automatically by the code to ensure accuracy of the particle pusher; this applies a scaling factor to it.
* `select_multiplicity` (optional): to facilitate comparisons with theory, select only those showers with the desired number of daughter particles when creating output.
* `lcfa` (optional, default = `false`): if `true`, use rates calculated in the locally constant, crossed fields approximation to model QED processes.

## laser

* `a0`: the laser strength parameter, normalized amplitude, etc.
* `wavelength`: of the carrier wave, in metres, or
* `omega`: the equivalent photon energy, in joules. The conversion constants `eV` etc are provided for convenience.
* `waist` (optional): if specified, the laser pulse will be focused to a spot size of `waist`, which defines the radius at which the intensity falls to 1/e^2 of its maximum value. Otherwise the laser is modelled as a plane wave.
* `fwhm_duration` (if `waist` is specified): if focusing, the laser pulse has a Gaussian temporal profile in intensity, with the specified duration (full width at half max) in seconds.
* `n_cycles` (if `waist` is not specified): if not focusing, the laser pulse has a cos^2 envelope in electric field, with total duration equal to the given number of wavelengths.
* `chirp_coeff` (optional, ignored if `waist` is specified): specifies `b`, the chirp coefficient, which appears in the total phase `ϕ + b ϕ^2` of the laser carrier wave. A positive `b` leads to an instantaneous frequency that increases linearly from head to tail.

## beam

* `ne`: number of primary electrons.
* `gamma`: the mean Lorentz factor.
* `sigma` (optional, default = `0.0`): the standard deviation of the electron Lorentz factors, set to zero if not specified.
* `radius`: if a single value is specified, the beam is given a cylindrically symmetric Gaussian charge distribution, with specified standard deviation in radius (metres). The distribution is set explicitly if a tuple of `[radius, dstr]` is given. `dstr` may be either `normally_distributed` (the default) or `uniformly_distributed`. In the latter case, `radius` specifies the maximum, rather than the standard deviation.
* `length` (optional, default = `0.0`): standard deviation of the (Gaussian) charge distribution along the beam propagation axis (metres)
* `collision_angle` (optional, default = `0.0`): angle between beam momentum and laser axis in radians, with zero being perfectly counterpropagating; the constant `degree` is provided for convenience.

## output

All output is written to the directory where the input file is found.

* `ident` (optional, default = no prefix): prepends a identifier string to the filenames of all produced output.
* `min_energy` (optional, default = `0.0`): if specified, discard secondary particles below a certain energy before creating the output distributions.
* `electron`: list of specifiers, each of which should correspond to a distribution function. For example, `x:px` requests the distribution of the x coordinate and the corresponding momentum component. Each separate output is written to its own FITS file.
* `photon`: as above.

## stats

If specified, writes aggregated statistical information about the particle final state distributions to a file called 'stats.txt' (with the appropriate identifier prefix, if specified in [output](#output)) in the same directory as the input file.

* `electron` (optional): list of specifiers
* `photon` (optional): list of specifiers

Each specifier must be one of:

* ``op var[`weight]``
* ``op var[`weight] in (min; max)``
* ``op var[`weight] for var2 in (min; max)``

where `op` is one of `total`, `fraction`, `mean`, `variance`, `minimum` and `maximum` and `var` is a desired output (`px`, the x-component of momentum, for example). The range of values to be used can be specified by `var`, or another output entirely, `var2`. Both `min` and `max` can be arbitrary mathematical expressions, using values given in the [constants](#constants) block, or `auto`, in which case the range is detected automatically. The contribution of each particle to the statistic is either its weight (i.e. number) or may given in terms of another variable.

For example: `mean energy` computes the average of the particle energy; ``variance angle_x`energy`` computes the energy-weighted variance of the angle between the particle momentum and the x axis; `mean px in (1.0; auto)` computes the average px for all particles that have px greater than 1; `total number for px in (1.0; 2.0)` calculates the number of particles with momentum component between the specified bounds.

## constants

Everywhere an integer or floating-point number is requested in the input file, a named value may be given instead, provided that its value is specified in this section.

For example, `gamma: eta * 0.510999 / (2.0 * 1.5498e-6)` in the [beam](#beam) section would be accepted provided that `eta: 0.1` was given. Named constants can themselves be mathematical expressions, but they cannot depend on each other, or themselves.

## Maths parser

The code makes use of [meval](https://crates.io/crates/meval) when parsing the input file. In addition to the functions and constants this crate provides, opal provides:

* the physical constants `me`, `mp`, `c`, `e`: the electron mass, proton mass, speed of light and elementary charge, respectively, all in SI units.
* the conversion constants `eV`, `keV`, `MeV`, `GeV`, `femto`, `pico`, `nano`, `milli`.
