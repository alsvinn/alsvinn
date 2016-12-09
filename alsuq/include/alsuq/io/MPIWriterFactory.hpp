#pragma once
#include "alsfvm/io/WriterFactory.hpp"
#include <mpi.h>
namespace alsuq { namespace io { 

    class MPIWriterFactory : alsfvm::io::WriterFactory {
    public:

        MPIWriterFactory(const std::vector<std::string>& groupNames,
                         size_t groupIndex,
                         MPI_Comm mpiCommunicator,
                         MPI_Info mpiInfo);


        alsfvm::shared_ptr<alsfvm::io::Writer>
        createWriter(const std::string &name, const std::string &baseFilename);

    private:
         std::vector<std::string> groupNames;
         size_t groupIndex;
         MPI_Comm mpiCommunicator;
         MPI_Info mpiInfo;
    };
} // namespace io
} // namespace alsuq
