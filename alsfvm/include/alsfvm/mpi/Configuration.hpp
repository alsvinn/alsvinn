#pragma once
#include "alsfvm/types.hpp"
#include <mpi.h>
namespace alsfvm { namespace mpi { 

    class Configuration {
    public:
        Configuration(MPI_Comm communicator,
                      const std::string& platform = "cpu");

        MPI_Comm getCommunicator();

        int getNodeNumber() const;

        int getNumberOfNodes() const;

        MPI_Info getInfo();
        std::string getPlatform() const;

    private:
        MPI_Comm communicator;

        int nodeNumber;
        int numberOfNodes;
        MPI_Info info;
        const std::string platform ="cpu";
    };

    typedef alsfvm::shared_ptr<Configuration> ConfigurationPtr;
} // namespace mpi
} // namespace alsfvm
