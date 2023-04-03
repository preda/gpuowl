Build

GpuOwl can be built using Make or Meson.


1. Build using Make

Simply run "make" in the base folder (the one containing the file "Makefile").
All the files produced by the build, including the executable "gpuowl", are written to the folder "build".


2. Build using Meson

Starting in the base folder (the one containing the file "meson.build" and "Makefile"), create an empty build folder;
cd to the build folder, and invoke meson pointing it to the base folder. Next run *ninja* to actually build.
Example:

mkdir mybuild
cd mybuild
meson ..
ninja



Roundoff error checking

Roundoff error (ROE) should never reach 0.5. It is possible to log the ROE to see how close it
is to the danger level (0.5). This is controlled with the -use options ROE1 and ROE2. ROE1 prints
the roundoff produced in carryFused, while ROE2 prints the roundoff produced in carryA.

Examples:
./gpuowl -use ROE1
./gpuowl -use ROE1,ROE2

Three values are printed, e.g. ROE=0.250 0.1859 10025

The first (0.250) is the maximum roundoff,
the second value (0.1859) is the standard deviation from zero (i.e. sqrt(sum(roundoff^2)/N)),
and the the third value (10025) is the number of carry iterations that are included in these stats.

Enabling ROE1 has about a 4% performance hit. Enabling ROE2 has almost no performance impact but is
also less informative.
