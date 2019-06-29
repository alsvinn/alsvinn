[![Build Status](https://travis-ci.org/alsvinn/alsvinn.svg?branch=master)](https://travis-ci.org/alsvinn/alsvinn) [![Documentation Status](https://readthedocs.org/projects/alsvinn/badge/?version=latest)](https://alsvinn.readthedocs.io/en/latest/?badge=latest)


![Alsvinn](https://github.com/alsvinn/alsvinn/raw/master/documentation/images/kh_small.png "Kelvin-Helmholtz simulation")
# Alsvinn 

Alsvinn is a toolset consisting of a finite volume simulator (FVM) and modules for uncertaintity quantifications (UQ).
All the major operations can be computed on either a multi-core CPU or an NVIDIA GPU (through CUDA).
It also supports cluster configurations consisting of either CPUs or GPUs. It exhibits excellent scaling.

Alsvinn is maintained by [Kjetil Olsen Lye](https://github.com/kjetil-lye/) at ETH Zurich. We want Alsvinn to be easy to use, so if you have issues compiling or running it, please don't hesitate to leave an issue.

Alsvinn is also [available as a Docker container](https://hub.docker.com/r/alsvinn/). See below for [how to run with Docker or Singularity without any installation needed.](#running-in-docker-or-singularity)

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

  * C++11 compiler (tested with clang and gcc)
  * [gtest](https://github.com/google/googletest) (optional)
  * [boost](https://www.boost.org/) (including boost-numpy)
  * [python](https://www.python.org) Tested with 3.7, should work with 2.7
  * [hdf5](https://support.hdfgroup.org/HDF5/),
  * [netcdf](https://www.unidata.ucar.edu/software/netcdf/)
  * [parallel-netcdf](https://trac.mcs.anl.gov/projects/parallel-netcdf) *NOTE*: This is *not* the same as building netcdf with parallel support
  * [doxygen](http://www.stack.nl/~dimitri/doxygen/) (optional)
  * [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone) (optional)

see [Installing necessary software](#installing-necessary-software) for more information.

## Cloning

This project uses a git submodule, the easiest way to clone the repository is by

    git clone --recursive https://github.com/alsvinn/alsvinn.git


## Compiling

Should be as easy as running (for advanced cmake-users: the location of the build folder can be arbitrary)

    mkdir build
    cd build
    cmake ..

note that you should probably run it with ```-DCMAKE_BUILD_TYPE=Release```, ie

    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release

### Compiling with CUDA

If you do not have a CUDA device on your computer, or if you do not have CUDA installed, you should run CMAKE with the ```-DALSVINN_USE_CUDA=OFF``` option.

## Running tests

Before you try to run the simulations, it's probably a good idea to validate that the build was successful by running the unittests. From the build folder, run

    ./test/library_test/alstest
    bash run_mpi_tests.sh

## Running alsvinn

The basic input of alsvinn are ```.xml```-files specifying the different options. You can view the different examples under ```examples```. The initial data is usually specified in a ```.py```-file (named in the xml file).

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


## Running in Docker or Singularity
To run alsvinn using Docker and get the output to the current directory, all you have to do is

    docker run --rm -v $(pwd):$(pwd) -w $(pwd) alsvinn/alsvinn_cuda /examples/kelvinhelmholtz/kelvinhelmholtz.xml

*NOTE* Replace ```alsvinn_cuda``` with ```alsvinn_cpu``` if you are running a CPU only setup.

*NOTE* You can also make your own configuration files and specify them instead of ```/examples/kelvinhelmholtz/kelvinhelmholtz.xml```

### Running with Singularity
On LSF systems with Singularity installed, Alsvinn can be run as 

    # Cluster WITHOUT GPUs
    bsub -R singularity singularity run -B $(pwd):$(pwd) \
         docker://alsvinn/alsvinn_cpu 
         /examples/kelvinhelmholtz/kelvinhelmholtz.xml
	 
    # Clusters WITH GPUs:
    bsub <other options to get GPU> \
         -R singularity singularity run --nv -B $(pwd):$(pwd) \
         docker://alsvinn/alsvinn_cuda \
         /examples/kelvinhelmholtz/kelvinhelmholtz.xml

Note that it is probably a good idea to pull the image first and then run the downloaded image, eg.

     bsub -R light -J downloading_image -R singularity \
         singularity pull docker://alsvinn/alsvinn_cuda
         
     bsub -R singularity -w "done(downloading_image)" \
         singularity run --nv -B $(pwd):$(pwd) \
         alsvinn_cuda.simg \
         /examples/kelvinhelmholtz/kelvinhelmholtz.xml
         
To run with MPI, we strongly recommend using MVAPICH2 (or any mpich ABI compatible implementation) on the cluster. On a lot of cluster, you would do something like:

    # Or whatver the mvapich2 module is called
    module load mvapich2
    bsub -R light -J downloading_image -R singularity \
         singularity pull docker://alsvinn/alsvinn_cuda
         
     bsub -R singularity -w "done(downloading_image)" \
         mpirun -np <number of cores> singularity run --nv -B $(pwd):$(pwd) \
         alsvinn_cuda.simg  --multi-y <number of cores> \
         /examples/kelvinhelmholtz/kelvinhelmholtz.xml

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


## Installing necessary software

### Ubuntu 19.04

Simply run

    sudo apt-get update
    sudo apt-get install libnetcdf-mpi-dev libnetcdf-dev
    	cmake python3 python3-matplotlib python3-numpy python3-scipy git \
	libopenmpi-dev gcc g++ libgtest-dev libboost-all-dev doxygen \
        build-essential graphviz libhdf5-mpi-dev libpnetcdf-dev

if you want CUDA (GPU) support, you have to install the CUDA packages as well

    sudo apt-get install nvidia-cuda-dev nvidia-cuda-toolkit

#### Compiling with CUDA

Compile with

    cd alsvinn/
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DALSVINN_PYTHON_VERSION=3.7
    make

(If you are using a different python version, replace 3.7 with whatever python version you are using.)

#### Compiling without CUDA

Compile with

    cd alsvinn/
    mkdir build
    cd build
    cmake .. -DALSVINN_USE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release \
        -DALSVINN_PYTHON_VERSION=3.7
    make
    
(If you are using a different python version, replace 3.7 with whatever python version you are using.)

### Arch Linux
Pacman should have all needed packages, simply install

    pacman -S git cmake cuda g++ netcdf parallel-netcdf-openmpi hdf5-openmpi doxygen gtest boost python

### Manually installing parallel-netcdf

On some old distributions, you have to manually compile parallel-netcdf. You can [download the latest version from the webpage](https://trac.mcs.anl.gov/projects/parallel-netcdf/wiki/Download) and install using autotools. Notice that you should compile with the CFLAG ```-fPIC```. A quick way to install parallel-netcdf would be

    wget http://cucis.ece.northwestern.edu/projects/PnetCDF/Release/parallel-netcdf-1.9.0.tar.gz
    tar xvf parallel-netcdf-1.9.0.tar.gz
    cd parallel-netcdf-1.9.0
    export CFLAGS='-fPIC'
    ./configure --prefix=<some location>
    make install

remember to specify ```<some location>``` to ```-DCMAKE_PREFIX_PATH``` afterwards

# Using alsvinn as a library
While it is not recommend, there are [guides available on how to run alsvinn as a standalone library](library_examples/README.md).

