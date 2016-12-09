#include "alsuq/mpi/Config.hpp"
#include <mpi.h>
namespace alsuq { namespace mpi {

Config::Config(int argc, char **argv)
{

    MPI_Init(&argc, &argv);
    communicator = MPI_COMM_WORLD;
    info = MPI_INFO_NULL;
    MPI_Comm_size(communicator, &numberOfProcesses);
    MPI_Comm_rank(communicator, &rank);
}

MPI_Comm Config::getCommunicator()
{
    return communicator;
}

MPI_Info Config::getInfo()
{
    return info;
}

int Config::getNumberOfProcesses() const
{
    return numberOfProcesses;
}

int Config::getRank() const
{
    return rank;
}

}
}
