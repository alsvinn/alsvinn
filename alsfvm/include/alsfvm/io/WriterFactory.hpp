#pragma once
#include "alsfvm/io/Writer.hpp"


namespace alsfvm { namespace io { 

    //! This implements the abstract factory pattern
    //! The reason for doing this is that we sometimes want to use
    //! mpi writers, and sometimes not.
    class WriterFactory {
    public:
        virtual alsfvm::shared_ptr<Writer> createWriter(const std::string& name,
                                                     const std::string& baseFilename);
    };
} // namespace io
} // namespace alsfvm
