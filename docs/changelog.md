# Changelog

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