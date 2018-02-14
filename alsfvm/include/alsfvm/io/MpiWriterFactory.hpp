#pragma once
#include "alsfvm/io/WriterFactory.hpp"
#include "alsfvm/mpi/Configuration.hpp"

namespace alsfvm {
namespace io {

//! This implements the abstract factory pattern
//! The reason for doing this is that we sometimes want to use
//! mpi writers, and sometimes not.
class MpiWriterFactory : public WriterFactory {
public:
    MpiWriterFactory(mpi::ConfigurationPtr configuration);

    virtual alsfvm::shared_ptr<Writer> createWriter(const std::string& name,
        const std::string& baseFilename);

private:
    mpi::ConfigurationPtr configuration;
};
} // namespace io
} // namespace alsfvm
