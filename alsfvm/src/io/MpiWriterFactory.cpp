#include "alsfvm/io/MpiWriterFactory.hpp"
#include "alsfvm/io/NetCDFMPIWriter.hpp"

#include "alsutils/error/Exception.hpp"

namespace alsfvm { namespace io {

MpiWriterFactory::MpiWriterFactory(mpi::ConfigurationPtr configuration)
    :configuration(configuration)
{

}

alsfvm::shared_ptr<Writer> MpiWriterFactory::createWriter(const std::string &name,
                                                          const std::string &baseFilename)
{
    alsfvm::shared_ptr<Writer> writer;
    if (name == "netcdf") {
        writer.reset(new NetCDFMPIWriter(baseFilename, {""}, 0, true,
                                         configuration->getCommunicator(),
                                         configuration->getInfo()));
    } else {
        THROW("Unknown writer " << name << std::endl << "Does not have MPI support.");
    }

    return writer;
}

}
}
