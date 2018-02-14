#include "alsutils/mpi/mpi_types.hpp"

namespace alsutils {
namespace mpi {
template<>
const MPI_Datatype MpiTypes<double>::MPI_Real = MPI_DOUBLE;

template<>
const MPI_Datatype MpiTypes<float>::MPI_Real = MPI_FLOAT;
}
}
