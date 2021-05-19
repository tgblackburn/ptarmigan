# Changelog

## v0.7.0

In progress.

New:

* Finite bandwidth correction: optionally account for the effect of the laser pulse duration when sampling photon momenta.
* Electron-positron pair creation, via LMA and LCFA rates. `leading-order-only` feature, if enabled, replaces the LMA rate with a perturbative equivalent.
* Previous behaviour, i.e. no pair creation or photon tracking, is recovered by compiling with `no-pair-creation` feature.
* Photon primaries, with either Gaussian or bremsstrahlung energy spectra.

Fixed:

* Build failure caused by missing git info.

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
