# Loading custom particle beams

So, you have a particle beam that cannot be represented by Ptarmigan's standard input parameters?
Perhaps the six-dimensional phase space is highly correlated, or you have output from another code you want to use?

Creating an HDF5 file of particle data that contains the following datasets:

* `beam_axis`: a string that is one of `+x`, `-x`, `+z` or `-z`, indicating the beam propagation direction.
* `config/unit/momentum`: a UTF-8 formatted string that gives the units of the four-momentum.
Ptarmigan will recognise `kg/m/s`, as well as `MeV/c` and `GeV/c` (with or without the `/c`).
* `config/unit/position`: a UTF-8 formatted string that gives the units of the four-position.
Ptarmigan will recognise `um`, `micron`, `mm` and `m`.
* `final-state/{particle}/weight`: an N⁢ × 1 array of doubles that gives the particle weights, i.e. the number of real particles represented.
* `final-state/{particle}/momentum`: an N⁢ × 4 array of doubles that gives the particle four-momenta $(\gamma m c, \gamma m \vec{v})$ or $(E/c, \vec{p})$.
* `final-state/{particle}/position`: an N⁢ × 4 array of doubles that gives the particle four-positions $(c t, \vec{r})$.

and optionally:

* `final-state/{particle}/polarization`: an N⁢ × 4 array of doubles that gives the particle Stokes parameters $(1, S_1, S_2, S_3)$.
$S_1$ and $S_2$ represent the degree of linear polarization and $S_3$ the degree of circular polarization.

will allow it to be imported by Ptarmigan.

`{particle}` should be one of `electron`, `positron` or `photon`.
The species that is to be imported is specified in the Ptarmigan input file.