#include "alsfvm/mpi/Request.hpp"
#include "alsutils/mpi/safe_call.hpp"

namespace alsfvm {
namespace mpi {



Request::Request() {

}

void Request::wait() {
    MPI_SAFE_CALL(MPI_Wait(&request, MPI_STATUS_IGNORE));
}

Request::~Request() {
    if (request != NULL) {
        this->wait();
    }
}

}
}
