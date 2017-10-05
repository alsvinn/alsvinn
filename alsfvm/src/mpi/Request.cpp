#include "alsfvm/mpi/Request.hpp"
#include "alsutils/mpi/safe_call.hpp"

namespace alsfvm { namespace mpi {



Request::Request()
{

}

void Request::wait()
{
    MPI_Status status;
    MPI_SAFE_CALL(MPI_Wait(&request, &status));
}

Request::~Request()
{
    this->wait();
}

}
}
