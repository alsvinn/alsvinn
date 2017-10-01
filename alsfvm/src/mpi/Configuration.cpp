#include "alsfvm/mpi/Configuration.hpp"

namespace alsfvm { namespace mpi {

Configuration::Configuration(MPI_Comm communicator)
    : communicator(communicator)
{
    MPI_Comm_rank(communicator, &nodeNumber);

}

MPI_Comm Configuration::getCommunicator()
{
    return communicator;
}

int Configuration::getNodeNumber() const
{

}

}
}
