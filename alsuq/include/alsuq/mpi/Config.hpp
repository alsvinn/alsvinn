#pragma once
#include <mpi.h>
namespace alsuq { namespace mpi { 

    class Config {
    public:
        Config();

        MPI_Comm getCommunicator();
        MPI_Info getInfo();

        int getNumberOfProcesses() const;
        int getRank() const;
    private:
        int numberOfProcesses;
        int rank;
        MPI_Comm communicator;
        MPI_Info info;
    };
} // namespace mpi
} // namespace alsuq
