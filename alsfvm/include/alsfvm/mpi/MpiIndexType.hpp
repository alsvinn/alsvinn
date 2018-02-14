#pragma once
#include <mpi.h>
#include <vector>
#include "alsfvm/types.hpp"

namespace alsfvm {
namespace mpi {

class MpiIndexType {
    private:
        MpiIndexType(int numberOfBlocks,
            const std::vector<int>& blockLengths,
            const std::vector<int>& offsets,
            MPI_Datatype datatype
        );
    public:
        static alsfvm::shared_ptr<MpiIndexType> makeInstance(int numberOfBlocks,
            const std::vector<int>& blockLengths,
            const std::vector<int>& offsets,
            MPI_Datatype datatype);
        ~MpiIndexType();

        MPI_Datatype indexedDatatype();
    private:
        MPI_Datatype indexedDatatypeMember;
};

typedef alsfvm::shared_ptr<MpiIndexType> MpiIndexTypePtr;
} // namespace mpi
} // namespace alsfvm
