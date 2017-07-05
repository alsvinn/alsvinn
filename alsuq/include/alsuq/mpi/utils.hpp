#pragma once
#include "alsutils/error/Exception.hpp"
#include <mpi.h>

#define MPI_SAFE_CALL(x) { \
    if (x != MPI_SUCCESS) { \
        THROW("MPI failured with \n\t"<< #x); \
    }\
}
