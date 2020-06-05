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

## beam

* `ne`: number of primary electrons.
* `gamma`: the mean Lorentz factor.
* `sigma` (optional, default = `0.0`): the standard deviation of the electron Lorentz factors, set to zero if not specified.
* `radius`: the beam has a cylindrically symmetric Gaussian charge distribution, with specified standard deviation in radius (metres)...
* `length` (optional, default = `0.0`): and length (metres)

## output

All output is written to the directory where the input file is found.

* `ident` (optional, default = no prefix): prepends a identifier string to the filenames of all produced output.
* `min_energy` (optional, default = `0.0`): if specified, discard secondary particles below a certain energy before creating the output distributions. 
* `electron`: list of specifiers, each of which should correspond to a distribution function. For example, `x:px` requests the distribution of the x coordinate and the corresponding momentum component. Each separate output is written to its own FITS file.
* `photon`: as above.

## constants

Everywhere an integer or floating-point number is requested in the input file, a named value may be given instead, provided that its value is specified in this section.

For example, `gamma: eta * 0.510999 / (2.0 * 1.5498e-6)` in the [beam](#beam) section would be accepted provided that `eta: 0.1` was given. Named constants can themselves be mathematical expressions, but they cannot depend on each other, or themselves.

## Maths parser

The code makes use of [meval](https://crates.io/crates/meval) when parsing the input file. In addition to the functions and constants this crate provides, opal provides:

* the physical constants `me`, `mp`, `c`, `e`: the electron mass, proton mass, speed of light and elementary charge, respectively, all in SI units.
* the conversion constants `eV`, `keV`, `MeV`, `GeV`, `femto`, `pico`, `nano`, `milli`.
