#include "alsfvm/mpi/Configuration.hpp"

namespace alsfvm { namespace mpi {

Configuration::Configuration(MPI_Comm communicator,
                             const std::string& platform)
    : communicator(communicator), platform(platform)
{
    MPI_Comm_rank(communicator, &nodeNumber);
    MPI_Comm_size(communicator, &numberOfNodes);
    info = MPI_INFO_NULL;
}

MPI_Comm Configuration::getCommunicator()
{
    return communicator;
}

int Configuration::getNodeNumber() const
{
    return nodeNumber;
}

int Configuration::getNumberOfNodes() const
{
    return numberOfNodes;
}

MPI_Info Configuration::getInfo()
{
    return info;
}

std::string Configuration::getPlatform() const
{
    return platform;
}

}
                 }
