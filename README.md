Alsvinn - The fast Finite Volume Simulator with support for Uncertainty Quantifications
------------------------

Alsvinn is a toolset consisting of a finite volume simulator (alsfvm) and modules in python for uncertainity quantification (UQ).

## Requirements

  * C++11 compiler (tested with clang, gcc and MSVC-12.0)
  * gtest (optional)
  * hdf5
  * doxygen (optional)
  * cuda (optional)

## Notes on Windows

You will need to download ghostcript in order for doxygen to work. Download from 

     https://code.google.com/p/ghostscript/downloads/detail?name=gs910w32.exe&can=2&q=

Sometimes cmake finds libhdf5 as the dll and not the lib-file. Check that the HDF5_C_LIBRARY is set to hdf5.lib in cmake

gtest should be built with "-Dgtest_force_shared_crt=ON" via cmake. This will ensure that there is no library compability issues. 

You will also want to compile a Debug AND Release version of Gtest, and set each library manually in Cmake. 

If you have installed Anaconda AND HDF5, make sure the right HDF5 version is picked (it is irrelevant which one you pick, 
but the two can intertwine in one or several of: include directories, libraries and Path (for dll).)



