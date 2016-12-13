#include "alsfvm/io/WriterFactory.hpp"
#include "alsfvm/io/HDF5Writer.hpp"

namespace alsfvm { namespace io {

alsfvm::shared_ptr<Writer> WriterFactory::createWriter(const std::string &name,
                                                    const std::string& baseFilename)
{
    alsfvm::shared_ptr<Writer> writer;
    if (name == "hdf5") {
        writer.reset(new HDF5Writer(baseFilename));
    } else {
        THROW("Unknown writer type " << name);
    }

    return writer;
}

}
}