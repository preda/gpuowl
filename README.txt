PRPLL "purrple cat[egorizer]"

Build
-----

To build PRPLL use "make".

In the gpuowl project directory (where the file Makefile is located) run make.
This will produce a file "prpll" in the build-debug or build-release subdirectory.

Use "make" to do a release build in the "build-release" subdirectory.
Use "make DEBUG=1" to produce a debug build in the "build-debug" subdirectory.
Use "make exe" for a Windows build.
Use "make clean" to remove the "build-debug" and "build-release" directories.

Build example:

$ git clone git@github.com:preda/gpuowl.git
$ cd gpuowl
$ make
$ build-release/prpll -h


Run
----

1. run "clinfo" and see whether it finds any OpenCL devices. If clinfo does not find any devices, 
you need to fix your OpenCL installation first.

2. run "prpll -h", and verify that it displays a list of devices towards the end.
