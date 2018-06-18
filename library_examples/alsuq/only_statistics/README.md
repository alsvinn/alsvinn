Using only the statistics module of alsuq
===========================================
This example is meant as a minimal example of how to use only the statistics component of ```alsvinn::alsuq```.

# Prerequisites
First you need too build alsvinn, see the documentation.


# Building

This should be straightforward:

    mkdir build
    cd build
    cmake .. -DCMAKE_PREFIX_PATH=<path to either alsvinn/build or alsvinn-install-path> -DCMAKE_BUILD_TYPE=Release
    make

