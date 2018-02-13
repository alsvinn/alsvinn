#pragma once
#include "alsutils/types.hpp"
#include <mpi.h>
namespace alsutils {
namespace mpi {
template<class T>
struct MpiTypes {
  static const MPI_Datatype MPI_Real;
};



}
}
