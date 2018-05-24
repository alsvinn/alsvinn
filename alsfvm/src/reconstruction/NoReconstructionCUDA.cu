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

#include "alsfvm/reconstruction/NoReconstructionCUDA.hpp"
#include "alsfvm/cuda/cuda_utils.hpp"

namespace alsfvm {
namespace reconstruction {
size_t NoReconstructionCUDA::getNumberOfGhostCells()  {
    return 1;
}


void NoReconstructionCUDA::performReconstruction(const volume::Volume&
    inputVariables,
    size_t direction,
    size_t indicatorVariable,
    volume::Volume& leftOut,
    volume::Volume& rightOut, const ivec3& start,
    const ivec3& end) {

    for (size_t var = 0; var < inputVariables.getNumberOfVariables(); ++var) {
        CUDA_SAFE_CALL(cudaMemcpyAsync(leftOut.getScalarMemoryArea(var)->getPointer(),
                inputVariables.getScalarMemoryArea(var)->getPointer(),
                leftOut.getScalarMemoryArea(var)->getSize() * sizeof(real),
                cudaMemcpyDeviceToDevice));
        CUDA_SAFE_CALL(cudaMemcpyAsync(rightOut.getScalarMemoryArea(var)->getPointer(),
                inputVariables.getScalarMemoryArea(var)->getPointer(),
                rightOut.getScalarMemoryArea(var)->getSize() * sizeof(real),
                cudaMemcpyDeviceToDevice));
    }
}

}
}
