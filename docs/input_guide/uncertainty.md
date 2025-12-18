# Uncertainty tracking

### Theory uncertainties

* `lcfa` (optional, default = `0.0`): if enabled, output spectra will include an estimate for the
uncertainty due to the use of the LCFA, at the requested number of standard deviations.
Requires Ptarmigan to have been compiled with the feature `uncertainty-tracking`.
The code will run slower if this feature has been activated.