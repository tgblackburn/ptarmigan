# Generating a particle beam

### Basic information

* `n`: number of primary particles. `ne` is also accepted.
* `species` (optional, default = `electron`): primary particle type, must be one of `electron`, `photon` or `positron`.
* `n_real` (optional, default = `n`): if specified, weight each primary particle such that the whole ensemble represents the given number of real particles, or
* `charge` (optional): if specified, weight each primary electron such that the bunch has this total charge. Include a factor of the elementary charge `e` to get a specific number of electrons.
It is not necessary to provide both `n_real` and `charge`.
`charge` can also be used to assign a weight to primary *photons*, by assuming that each photon has a fictitious elementary charge, but prefer `n_real` for this purpose.

### Energy spectrum

By default, Ptarmigan assumes the primaries to have normally distributed momenta.
The mean and standard deviation of the energy in units of the electron rest energy (i.e. the Lorentz factor) is controlled by:

* `gamma`: the mean Lorentz factor.
* `sigma` (optional, default = `0.0`): the standard deviation of the Lorentz factors.

and the transverse momenta by:

* `rms_divergence` (optional, default = `0.0`): if specified, the angles between particle initial momenta and the beam propagation axis are normally distributed, with given standard deviation.

Alternatively, you can specify a custom spectrum in a sub-section of `beam`:

* `spectrum`:
  * `function`: a mathematical expression for the spectrum as a function of `gamma`, **or**:
  * `file`: path to a plain-text file of new-line separated values of the spectrum, evaluated at evenly spaced points.
  * `min`: the minimum value of the Lorentz factor.
  * `max`: the maximum value of the Lorentz factor, **or**, if reading from a file:
  * `step` (optional): the spacing between sampling points.

Examples:

* A standard, Gaussian energy spectrum:
  ```yaml
  beam:
    # ...
    gamma: 1000 * MeV / (m * c^2)
    sigma: 100 * MeV / (m * c^2)
  ```

* A skewed energy spectrum, defined analytically.
Ptarmigan will look up named parameters (`mu`, `alpha` and `sigma`) in the [constants](input_guide/parsing.md) block.
  ```yaml
  beam:
    # ...
    spectrum:
      function: gauss(gamma, mu, sigma) * (1.0 + erf(alpha * (gamma - mu) / sigma))
      min: 100.0 * MeV / (m * c^2)
      max: 1500.0 * MeV / (m * c^2)
  ```

* A beam of bremsstrahlung photons:
  ```yaml
  beam:
    species: photon
    # ...
    spectrum:
      function: 4.0 * gamma_0 / (3.0 * gamma) - 4.0 / 3.0 + gamma / gamma_0 # thin-foil approximation
      min: gamma_cut
      max: gamma_0
  ```

* A skewed energy spectrum, but defined in an external file.
  ```yaml
  beam:
    # ...
    spectrum:
      file: spectrum.dat
      min: 100.0 * MeV / (m * c^2)
      step: 10.0 * MeV / (m * c^2) # or
      # max: 1500.0 * MeV / (m * c^2)
  ```
  where `spectrum.dat` contains
  ```txt
  0.15911901743645537
  0.1682733868638959
  0.17784323491877269
  0.18783989159800885
  0.19827450502149477
  ...
  ```

### Spatial distribution

* `radius` (optional, default = `0.0`): if a single value is specified, the beam is given a cylindrically symmetric Gaussian charge distribution, with specified standard deviation in radius (metres). The distribution is set explicitly if a tuple of `[radius, dstr]` is given. `dstr` may be either `normally_distributed` (the default) or `uniformly_distributed`. In the latter case, `radius` specifies the maximum, rather than the standard deviation.
The distribution (if normal) may be optionally truncated by specifying `[radius, normally_distributed, max_radius]`.
* `length` (optional, default = `0.0`): standard deviation of the (Gaussian) charge distribution along the beam propagation axis (metres)
* `energy_chirp` (optional, default = `0.0`): if specified, introduces a correlation of the requested magnitude between the particle's energy and its longitudinal offset from the beam centroid. A positive chirp means that the head of the beam (which hits the laser first) has higher energy than the tail. The specified value must be between -1 and +1.

### Spin and polarization

* `stokes_pars` (optional, default = `[0.0, 0.0, 0.0]`): specifies the primary particles' polarization in terms of the three Stokes parameters `S_1`, `S_2` and `S_3` (equiv. `Q`, `U` and `V`).
The basis is defined with respect to the `x`-`z` plane and the particle velocity:
`S_1` is associated with linear polarization along x (`+1.0`) or y (`-1.0`); `S_2` with linear polarization at 45 degrees to these axes; and `S_3` to the degree of circular polarization.
For example, `[1.0, 0.0, 0.0]` loads particles that are polarized parallel to the laser electric field (if `laser:polarization` is `linear`).
The default behaviour is to assume that the particles are unpolarized.

### Collision parameters

* `collision_angle` (optional, default = `0.0`): angle between beam momentum and laser axis in radians, with zero being perfectly counterpropagating; the constant `degree` is provided for convenience.
* `offset` (optional, default = `[0.0, 0.0, 0.0]`): introduces an alignment error between the particle beam and the laser pulse, as defined by the location of the beam centroid at the time when the peak of the laser pulse passes through focus.
The offsets are defined with respect to the beam propagation axis: the first two components are perpendicular to this axis and the third is parallel to it.
For example, if the offset is `[0.0, 0.0, delta > 0]` and the collision angle is `0.0`, the peak of the laser reaches the focal plane before the beam centroid does; the collision, while perfectly aligned in the perpendicular directions, is delayed by time `delta/(2c)`.
