# Control options

### Physics

* `radiation_reaction` (optional, default = `true`): set to `false` to disable electron/positron recoil on photon emission.
* `pair_creation` (optional, default = `true`): set to `false` to disable photon tracking and electron-positron pair creation.
* `pol_resolved` (optional, default = `false`): enable the use of photon-polarization-resolved rates, which improves simulation accuracy by approximately 20%.
* `classical` (optional, default = `false`): use the classical photon emission rate. If `radiation_reaction` is enabled, electrons and positrons lose energy smoothly, following the Landau-Lifshitz equation. Disables pair creation unless otherwise specified.
A modified classical model can be chosen by setting `classical` to `gaunt_factor_corrected`.
In this model the instantaneous radiated power is reduced by the Gaunt factor g(χ) and the upper bound on the photon spectrum corrected to the electron energy.
This option is only available under the LCFA.
* `lcfa` (optional, default = `false`): if `true`, use rates calculated in the locally constant, crossed fields approximation to model QED processes.
* `bandwidth_correction` (optional, default = `false`, ignored if `lcfa: true`): if `true`, correct the photon momentum sampling algorithm to account for the laser pulse's finite bandwidth. Has no effect if LCFA rates are selected.

### Numerics

* `dt_multiplier` (optional, default = `1.0`): the size of the timestep is set automatically by the code to ensure accuracy of the particle pusher; this applies a scaling factor to it.
* `increase_pair_rate_by` (optional, default = `1.0`): if specified, increases the pair creation rate, while decreasing the weight of any created electrons and positrons, by the same factor. This helps resolve the positron spectrum when the total probability is much smaller than 1/N, where N is the number of primary particles. A setting of `auto` will be replaced by a suitable default value, as determined from the laser amplitude and particle energy. In principle, an arbitrarily large increase may be specified, because the code automatically adjusts it if the probability per timestep becomes too large. However, this will mean that a very large number of (low-weight) electrons and positrons will be generated and tracked.
* `rng_seed` (optional, default = `0`): an unsigned integer that, if specified, is used as the basis for seeding the PRNG.

### Tracking

* `track_secondaries` (optional, default = `true`): set to `false` to disable tracking of any secondary particles that are created.
* `select_multiplicity` (optional): to facilitate comparisons with theory, select only those showers with the desired number of daughter particles when creating output.
This option is not compatible with an increase in the pair creation rate.
* `stop_at_time` (optional): if specified, stops tracking at the given instant of time. Otherwise, the simulation tracks particles until they have travelled through the entire laser pulse. Time zero corresponds to the point at which the peak of the laser passes through the focal plane.
Specify `auto` in order to force tracking to continue until all particles have the same time coordinate.