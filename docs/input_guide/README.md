# Creating a Ptarmigan input file

Ptarmigan takes as its single argument the path to a YAML file describing the input configuration.
This file is divided into sections that set up the initial particle distributions,
externally injected electromagnetic fields, the desired physical processes, and what output to produce.

* The [`control`](control.md) block defines the basic properties of the simulation:
what physics processes to include, which approximations to employ, and which particles to track and for how long.

* The [`laser`](laser.md) block defines the high-intensity laser pulse that takes part in the collision.

* The incident particle beam is defined in the section `beam`.
There are two ways to generate the particles of the incident beam in Ptarmigan.
Either the spectral and spatial properties of the beam can be given in the input file,
and the particles are then pseudorandomly sampled from these distributions,
or the particles can be imported from an external binary file.
Find out more about [generating a particle beam](beam_generation.md) and [loading a particle beam](beam_loading.md).

* The [`output`](output.md) and [`stats`](output.md#summary-statistics) blocks define what output
Ptarmigan will produce, including raw data, distribution functions, and summary statistics.

* The `constants` block contains user-specified named variables, which can be used anywhere a numerical value is
required in the input file. More information about parsing of numerical values [is given here](parsing.md).

A complete list of the possible keys [is given here](list_of_keys.md).
