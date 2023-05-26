# Changelog

## v1.3.2

2023-05-23

Fixed:

* Failure to set automatic pair rate increase when out of tabulated range.

## v1.3.1

2023-05-17

Added:

* Photon-polarization dependence of pair creation rates.
* Input file parsing: variables in the `constants` block can depend on previously defined variables.
* Evaluate and print math expressions in stats file (PR #46).
* Circular statistics (PR #47).

Fixed:

* Vulnerabilities reported by Dependabot.
* Warning about deprecated code in nom 1.2.4.
* Rate ceiling at large harmonic order (NLC, LP)

## v1.2.1

2023-01-23

Added:

* Looping over a range of a0s (PR #45).

Fixed:

* Incorrect sampling of LCFA pair-creation spectrum (evident only for chi > 2).

## v1.1.0

2023-01-13

Added:

* Modified (quantum-corrected) classical RR.
* Faster photon generation in CP backgrounds, under LMA.

Fixed:

* Error messages.

## v1.0.0

2022-12-01

Added:

* Classical radiation reaction.
* Choice of laser temporal envelope.
* Range cuts to distribution output.
* More detailed naming of FITS output files.

Fixed:

* Upgraded to mpi v0.6.
* Compile error for HDF5 versions < 1.10.
* Incorrect unit conversion.

Removed:

* Requirement to specify beam radius (default is zero).

## v0.11.3

2022-07-25

Fixed:

* Sanity check during generation of NLC photon momentum.

## v0.11.2

2022-07-13

Fixed:

* Empty (zero-length) datasets are not skipped when generating HDF5 output.

## v0.11.1

2022-07-07

Added:

* Optional discard of photons that have not pair-created, controlled by input file setting `discard_background`.

Fixed:

* Momentum sampling under LMA at finite collision angle.

## v0.11.0

2022-06-23

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
