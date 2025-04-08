# Mathematical expressions and parsing

Everywhere an integer or floating-point number is requested in the input file, a named value may be given instead,
provided that its value is specified in the `constants` section.

For example, `gamma: gamma0` in the [beam](beam_generation.md#energy-spectrum) section would be accepted provided that `gamma0` was appropriately defined.

Entries in the constants block can be plain numbers, mathematical expressions, or functions of previously defined entries. For example,
```yaml
constants:
  a0: 15.0
  gamma0: 1000.0 / 0.511
  max_angle: atan(a0 / gamma0)
```
would make the numerical values `a0 = 15.0`, `gamma = 1956.95` and `max_angle = 7.6649e-3` available throughout the input file.
The block is parsed once, in order, so variables cannot be forward-defined.
YAML syntax requires that keys are unique: repeated variables will raise an error.


The code makes use of [evalexpr](https://crates.io/crates/evalexpr) when parsing the input file. In addition to the functions and constants this crate provides, Ptarmigan provides:

* the physical constants `me`, `mp`, `c`, `e`: the electron mass, proton mass, speed of light and elementary charge, respectively, all in SI units.
* the conversion constants `eV`, `keV`, `MeV`, `GeV`, `femto`, `pico`, `nano`, `micro`, `milli` and `degree`.