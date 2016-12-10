#include "alsuq/io/MPIWriterFactory.hpp"
#include "alsfvm/io/HDF5MPIWriter.hpp"
namespace alsuq { namespace io {

MPIWriterFactory::MPIWriterFactory(const std::vector<std::string> &groupNames, size_t groupIndex,
                                   bool createFile,
                                   MPI_Comm mpiCommunicator, MPI_Info mpiInfo)

    : groupNames(groupNames), groupIndex(groupIndex), createFile(createFile),
      mpiCommunicator(mpiCommunicator),
      mpiInfo(mpiInfo)
{

}

alsfvm::shared_ptr<alsfvm::io::Writer> MPIWriterFactory::createWriter(const std::string &name,
                                                                      const std::string &baseFilename)
{
    alsfvm::shared_ptr<alsfvm::io::Writer> writer;
    if (name == "hdf5") {
        writer.reset(new alsfvm::io::HDF5MPIWriter(baseFilename, groupNames,
                                                   groupIndex, createFile, mpiCommunicator,
                                                   mpiInfo));
    } else {
        THROW("Unknown writer " << name);
    }

    return writer;
}

}
}
