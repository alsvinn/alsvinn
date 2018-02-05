#pragma once
#include "alsutils/types.hpp"
#include <mpi.h>
namespace alsutils { namespace mpi {

    class Configuration {
    public:
        Configuration(MPI_Comm communicator,
                      const std::string& platform = "cpu");

        MPI_Comm getCommunicator();

        int getRank() const;

        int getNumberOfProcesses() const;

        MPI_Info getInfo();
        std::string getPlatform() const;

        //! Essentially maps to MPI_Comm_split. See tutorial here:
        //! http://mpitutorial.com/tutorials/introduction-to-groups-and-communicators/
        alsfvm::shared_ptr<Configuration> makeSubConfiguration(int color,
                                                               int newRank) const;

    private:
        MPI_Comm communicator;

        int nodeNumber;
        int numberOfNodes;
        MPI_Info info;
        const std::string platform ="cpu";
    };

    typedef alsfvm::shared_ptr<Configuration> ConfigurationPtr;
} // namespace mpi
} // namespace alsutils
