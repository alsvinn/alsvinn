#pragma once
#include <mpi.h>
#include <vector>

namespace alsfvm { namespace mpi { 

    class MpiIndexType {
    public:
        MpiIndexType(int numberOfBlocks,
                     const std::vector<int>& blockLengths,
                     const std::vector<int>& offsets,
                     MPI_Datatype datatype
                     );
        ~MpiIndexType();

        MPI_Datatype indexedDatatype();
    private:
        MPI_Datatype indexedDatatypeMember;
    };
} // namespace mpi
} // namespace alsfvm
