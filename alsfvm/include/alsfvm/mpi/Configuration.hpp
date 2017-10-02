#pragma once
#include "alsfvm/types.hpp"
#include <mpi.h>
namespace alsfvm { namespace mpi { 

    class Configuration {
    public:
        Configuration(MPI_Comm communicator);

        MPI_Comm getCommunicator();

        int getNodeNumber() const;

        MPI_Info getInfo();
    private:
        MPI_Comm communicator;

        int nodeNumber;
        MPI_Info info;
    };

    typedef alsfvm::shared_ptr<Configuration> ConfigurationPtr;
} // namespace mpi
} // namespace alsfvm
