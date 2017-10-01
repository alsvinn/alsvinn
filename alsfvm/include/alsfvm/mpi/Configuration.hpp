#pragma once
#include "alsfvm/types.hpp"
#include <mpi.h>
namespace alsfvm { namespace mpi { 

    class Configuration {
    public:
        Configuration(MPI_Comm communicator);

        MPI_Comm getCommunicator();

        int getNodeNumber() const;
    private:
        MPI_Comm communicator;

        int nodeNumber;
    };

    typedef alsfvm::shared_ptr<Configuration> ConfigurationPtr;
} // namespace mpi
} // namespace alsfvm
