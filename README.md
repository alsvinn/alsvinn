Alsvinn - The fast Finite Volume Simulator with support for Uncertainty Quantifications
------------------------

Alsvinn is a toolset consisting of a finite volume simulator (alsfvm) and modules in python for uncertainity quantification (UQ).

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

    ./alsvinncli/alsuqcli <path to xml file>
 
 it has some options for mpi parallelization (for all options run ```alsuqcli/alsuqcli --help```). To run with MPI support, run eg
 
     mpirun -np <number of processes> ./alsvinncli/alsvinncli --multi-sample <number of procs in sample direction> path-to-xml.xml


## Notes on Windows

You will need to download ghostcript in order for doxygen to work. Download from 

     https://code.google.com/p/ghostscript/downloads/detail?name=gs910w32.exe&can=2&q=

Sometimes cmake finds libhdf5 as the dll and not the lib-file. Check that the HDF5_C_LIBRARY is set to hdf5.lib in cmake

gtest should be built with "-Dgtest_force_shared_crt=ON" via cmake. This will ensure that there is no library compability issues. 

You will also want to compile a Debug AND Release version of Gtest, and set each library manually in Cmake. 

If you have installed Anaconda AND HDF5, make sure the right HDF5 version is picked (it is irrelevant which one you pick, 
but the two can intertwine in one or several of: include directories, libraries and Path (for dll).)



