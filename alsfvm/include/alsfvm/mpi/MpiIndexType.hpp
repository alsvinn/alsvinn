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
