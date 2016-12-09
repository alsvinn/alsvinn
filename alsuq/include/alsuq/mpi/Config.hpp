#pragma once
#include <mpi.h>
namespace alsuq { namespace mpi { 

    class Config {
    public:
        Config(int argc, char** argv);

        MPI_Comm getCommunicator();
        MPI_Info getInfo();

        size_t getNumberOfProcesses() const;
        size_t getRank() const;
    private:
        size_t numberOfProcesses;
        size_t rank;
        MPI_Comm communicator;
        MPI_Info info;
    };
} // namespace mpi
} // namespace alsuq
