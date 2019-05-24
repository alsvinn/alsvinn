\page stats_alsuq_standalone Using the statistics
This example is meant as a minimal example of how to use only the statistics component of ```alsvinn::alsuq```.



# Prerequisites
First you need too build alsvinn, see the documentation.


# Building

This should be straightforward:

    mkdir build
    cd build
    cmake .. -DCMAKE_PREFIX_PATH=<path to either alsvinn/build or alsvinn-install-path> -DCMAKE_BUILD_TYPE=Release
    make

# Our goal

We want to compute the structure functions of "random noise". To this end, we want to use the statistics module of alsvinn.

# The code

## Necessary includes

\snippet src/main.cpp headers

## Making an instance of the statisticsFactory
We create a separate function to create the statitics:

\snippet src/main.cpp makeStructureFunction

To create statistics, we need to make a new instance of the statisticsFactory

\snippet src/main.cpp factoryInstance

then we setup the parameter struct

\snippet src/main.cpp parameters

notice how we use boost::property_tree for inserting parameters passed directly
to the statistics (p and numberOfH), while we set the other parameters to
the parameter struct directly

And lastly, we create and return the statistics.
\snippet src/main.cpp createStatistics

