# AVX_Noise

This project provides an AVX vectorized variant of a simplex noise generating
algorithm, with eight iterations of fractional brownian motion.

The resulting code is highly performant when compiled with preference to execution speed.


Currently, 1D and 2D simplex noise is supported. 3D and 4D will be supported soon.

In order to correctly generate noise, it is necessary to call the Noise::Seed
method after manually setting the Frequency, Amplitude, Lacunarity, and Persistence
to non-default values. Afterwards, you may call the Noise::getValue method to
generate simplex noise with eight iterations of fractional brownian motion.
