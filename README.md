![Alsvinn](https://github.com/kjetil-lye/alsvinn/raw/master/documentation/images/kh.png "Kelvin-Helmholtz simulation")
# Alsvinn - The fast FVM Simulator with support for UQ

Alsvinn is a toolset consisting of a finite volume simulator (FVM) and modules for uncertaintity quantifications (UQ). 
All the major operations can be computed on either a multi-core CPU or an NVIDIA GPU (through CUDA). 
It also supports cluster configurations cocnsisting of either CPUs or GPUs. It exhibits excellent scaling.

## Supported equations

  * The Compressible Euler Equations
  * The scalar Burgers' Equation
  * A scalar cubic conservation law
  * Buckley-Leverett
  
It is also possible to add new equations without issue (tutorial coming soon).

## Initial data

Initial data can easily be specified through a python script. For instance, the Sod Shock tube can be specified as
```python
if x < 0.0:
    rho = 1.
    ux = 0.
    p = 1.0
else:
    rho = .125
    ux = 0.
    p = 0.1
```

## Notable implemented initial data
While it's easy to implement new configurations, we already have a wide variety of configuraitons implemented, including:

   * Kelvin-Helmholtz instability
   * Richtmeyer-Meshkov instability
   * Sod shock tube
   * Cloudshock interaction
   * Shockvortex
   * Fractional Brownian motion
   


## Requirements

  * C++11 compiler (tested with clang, gcc and MSVC-12.0)
  * gtest (optional)
  * boost (including boost-numpy)
  * python 
  * hdf5, netcdf, parallel-netcdf
  * doxygen (optional)
  * cuda (optional)
 
## Compiling

Should be as easy as running (for advanced cmake-users: the location of the build folder can be arbitrary)

    mkdir build
    cd build
    cmake ..

note that you should probably run it with ```-DCMAKE_BUILD_TYPE=Release```, ie

    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    
    
## Running tests

Before you try to run the simulations, it's probably a good idea to validate that the build was successful by running the unittests. From the build folder, run

    ./test/alstest
    bash run_mpi_tests.sh

## Running alsvinn

The basic input of alsvinn are ```.xml```-files specifying the different options. You can view the different examples under ```alsvinncli/examples```. The initial data is usually specified in a ```.py```-file (named in the xml file).

### Deterministic runs

You make a deterministic run by running the ```alsvinncli``` utility. From the build folder, run

    ./alsvinncli/alsvinncli <path to xml file>
 
 it has some options for mpi parallelization (for all options run ```alsvinncli/alsvinncli --help```). To run with MPI support, run eg
 
     mpirun -np <number of processes> ./alsvinncli/alsvinncli --multi-x <number of procs in x direction> path-to-xml.xml

### UQ run

You make a UQ run by running the ```alsuqcli``` utility. From the build folder, run

    ./alsuqcli/alsuqcli <path to xml file>
 
 it has some options for mpi parallelization (for all options run ```alsuqcli/alsuqcli --help```). To run with MPI support, run eg
 
     mpirun -np <number of processes> ./alsuqcli/alsuqcli --multi-sample <number of procs in sample direction> path-to-xml.xml

## Output files

Most output is saved as a NetCDF file. These can easily be read in programming languages such as python.

## Python scripts

There is a simple python API for running alsvinn under the ```python``` folder. Check out the readme file there for more information.

## Note about Boost and Python

Be sure the python version you link with is the same as the the python version that was used with boost.
Usually, this is taken care of by using the libraries corresponding to the ```python``` executable (this is
especially true for CSCS Daint. On Euler you probably have to build boost with numpy support yourself).

## Note about GCC versions and CUDA

CUDA on Linux does at the moment not support GCC versions later than 6, therefore, to build with GPU support, you need to set the compiler to GCC-6.

After you have installed GCC-6 on your distribution, you can set the C/C++ compiler as

    cmake .. -DCMAKE_CXX_COMPILER=`which g++-6` -DCMAKE_C_COMPILER=`which gcc-6`

## Notes on Windows

You will need to download ghostcript in order for doxygen to work. Download from 

     https://code.google.com/p/ghostscript/downloads/detail?name=gs910w32.exe&can=2&q=

Sometimes cmake finds libhdf5 as the dll and not the lib-file. Check that the HDF5_C_LIBRARY is set to hdf5.lib in cmake

gtest should be built with "-Dgtest_force_shared_crt=ON" via cmake. This will ensure that there is no library compability issues. 

You will also want to compile a Debug AND Release version of Gtest, and set each library manually in Cmake. 

If you have installed Anaconda AND HDF5, make sure the right HDF5 version is picked (it is irrelevant which one you pick, 
but the two can intertwine in one or several of: include directories, libraries and Path (for dll).)



