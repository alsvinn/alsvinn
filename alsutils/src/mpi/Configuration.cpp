#include "alsutils/mpi/Configuration.hpp"
#include "alsutils/mpi/safe_call.hpp"

namespace alsutils { namespace mpi {

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

int Configuration::getRank() const
{
    return nodeNumber;
}

int Configuration::getNumberOfProcesses() const
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

alsfvm::shared_ptr<Configuration> Configuration::makeSubConfiguration(int color, int newRank) const
{
    MPI_Comm newCommunicator;
    MPI_SAFE_CALL(MPI_Comm_split(communicator, color, newRank, &newCommunicator));

    return ConfigurationPtr(new Configuration(newCommunicator, platform));
}

}
                 }
