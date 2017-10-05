#pragma once
#include "alsutils/types.hpp"
#include <mpi.h>
namespace alsutils {
namespace mpi {
template<class T>
struct MpiTypes {
};

template<>
struct MpiTypes<double> {
    static const constexpr MPI_Datatype MPI_Real = MPI_DOUBLE;
};


template<>
struct MpiTypes<float> {
    static const constexpr MPI_Datatype MPI_Real = MPI_FLOAT;
};

}
}
