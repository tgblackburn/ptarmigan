# Changelog

## v1.0.0

In progress

Added:

* Particle-beam transverse profile can be given by a truncated normal distribution.
* Parallelized HDF5 output.

Fixed:

* Building against OpenMPI v4.0+.

Removed:

* Plain-text output of complete particle data.
* `circular` as default choice of laser polarization; must be given explicitly.

## v0.10.1

2022-06-15

Fixed:

* Unnecessary warning about energy chirp when running with incident photons.
* Loss of precision in calculating ArcCos which led to particles || to primary being incorrectly discarded.

## v0.10.0

2022-06-10

Added:

* Photon-polarization-resolved emission rates, LMA (LP and CP) and LCFA.
* Nonlinear Breit-Wheeler pair creation for photons in LP lasers, using LMA.
* Output routines for Stokes parameters and polarization-weighted distributions.
* Energy chirping of incident particle beam.
* Early stopping at chosen time.

Fixed:

* Time-centering in particle push.
* Initialisation of beam with non-zero offset.
* Rate ceiling calculation for NLC (LP)

## v0.9.1

2022-04-13

Added:

* Nonlinear Compton scattering in linearly polarized backgrounds.
* Data attributes (units and description) in HDF-formatted output.
* Output specifiers `r_{x,y}` and option to filter out particles by angle.
* Runtime choice of file format for distribution output (plain text or FITS).
* More example input files.

Fixed:

* LMA photon sampling for non-zero collision angle.

Removed:

* Plain-text output of complete particle data (re-enabled by compiling with feature `enable-plain-text-dump`).
* `fits-output` feature. FITS output can be selected via the input file.

To be removed:

* `circular` as default choice of polarization

## v0.8.3

2021-11-02

Fixed:

* Output of birth a (normalised amplitude at particle creation point) under LCFA.

## v0.8.2

2021-08-19

Fixed:

* Overlapping RNG seeds.

## v0.8.1

2021-08-11

Added:

* Choice of units in FITS and HDF5 output.
* Offset between beam and laser when the latter reaches focus.

Fixed:

* Calculation of automatic pair rate increase when using LCFA

## v0.8.0

2021-07-12

Added:

* Output of particle and parent-particle IDs.
* Angularly resolved LCFA pair creation rate.

Fixed:

* Improved ponderomotive-force particle push.
* Pair rate increase automatically modified if too large.
* Stats output for positrons.

## v0.7.0

2021-06-01

New:

* Finite bandwidth correction: optionally account for the effect of the laser pulse duration when sampling photon momenta.
* Electron-positron pair creation, via LMA and LCFA rates. `leading-order-only` feature, if enabled, replaces the LMA rate with a perturbative equivalent.
* Previous behaviour, i.e. no pair creation or photon tracking, is recovered by running with `pair_creation: false` in the input file.
* Photon primaries, with either Gaussian or bremsstrahlung energy spectra.
* Automatic naming of output files.

Fixed:

* Build failure caused by missing git info.

Removed:

* `no-radiation-reaction` feature. Recoil on photon emission can now be disabled via the input file.

## v0.6.2

Fixed:

* Min-max finding in hgram
* Particle rotation in plain-text output

## v0.6.1

Adds new compile-time feature `cos2-envelope-in-3d` which switches the laser temporal envelope from the default Gaussian.

## v0.6.0

2021-02-24

New:

* Export of complete simulation data as HDF5, available when code compiled with feature `hdf5-output`.

Fixed:

* RNG seeding for different MPI tasks.

Removed:

* `write-velocity` feature, in advance of plain-text output itself.

## v0.5.2

2021-01-13

Incorporates uninitialized memory fix in linked-hash-map (dependency of yaml-rust, used by ptarmigan's input parsing), needed when compiling with rust >=1.48.

## v0.5.0

2021-01-08

Features:

* Classical propagation of an electron through a plane or focused electromagnetic wave.
* Nonlinear Compton scattering (photon emission), using rates calculated in the LMA (for a<sub>0</sub> < 10) or the LCFA.
* Plain-text or FITS output of particle distribution functions.
* Plain-text output of raw particle data (OUT formatted).

Compile-time options:

* `fits-output`: switches distribution function output to FITS format (requires cfitsio to be installed)
* `with-mpi`: enables parallel running of code
* `compensating-chirp`: if enabled, laser pulse is given a special chirp that compensates for the classical nonlinear redshift (1D only)
* `no-radiation-reaction`: disables recoil on photon emission (for testing only)
* `write-velocity`: write velocity, rather than momentum components, if raw particle output is requested
