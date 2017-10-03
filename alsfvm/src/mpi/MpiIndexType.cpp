#include "alsfvm/mpi/MpiIndexType.hpp"
#include "alsutils/log.hpp"

namespace alsfvm { namespace mpi {

MpiIndexType::MpiIndexType(int numberOfBlocks,
                           const std::vector<int> &blockLengths,
                           const std::vector<int> &offsets,
                           MPI_Datatype datatype)
{
    MPI_Type_indexed(numberOfBlocks, blockLengths.data(),
                     offsets.data(),
                     datatype,
                     &indexedDatatypeMember);
    MPI_Type_commit(&indexedDatatypeMember);
}

alsfvm::shared_ptr<MpiIndexType> MpiIndexType::makeInstance(int numberOfBlocks,
                                                            const std::vector<int> &blockLengths,
                                                            const std::vector<int> &offsets,
                                                            MPI_Datatype datatype)
{
     MpiIndexTypePtr pointer;
     pointer.reset(new MpiIndexType(numberOfBlocks, blockLengths,offsets, datatype));
     return pointer;
}

MpiIndexType::~MpiIndexType()
{
    MPI_Type_free(&indexedDatatypeMember);
}

MPI_Datatype MpiIndexType::indexedDatatype()
{
    return indexedDatatypeMember;
}

}
}
