/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "alsfvm/mpi/MpiIndexType.hpp"
#include "alsutils/log.hpp"

namespace alsfvm {
namespace mpi {

MpiIndexType::MpiIndexType(int numberOfBlocks,
    const std::vector<int>& blockLengths,
    const std::vector<int>& offsets,
    MPI_Datatype datatype) {
    MPI_Type_indexed(numberOfBlocks, (int*)blockLengths.data(),
        (int*)offsets.data(),
        datatype,
        &indexedDatatypeMember);
    MPI_Type_commit(&indexedDatatypeMember);
}

alsfvm::shared_ptr<MpiIndexType> MpiIndexType::makeInstance(int numberOfBlocks,
    const std::vector<int>& blockLengths,
    const std::vector<int>& offsets,
    MPI_Datatype datatype) {
    MpiIndexTypePtr pointer;
    pointer.reset(new MpiIndexType(numberOfBlocks, blockLengths, offsets,
            datatype));
    return pointer;
}

MpiIndexType::~MpiIndexType() {
    MPI_Type_free(&indexedDatatypeMember);
}

MPI_Datatype MpiIndexType::indexedDatatype() {
    return indexedDatatypeMember;
}

}
}
