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

#include "alsfvm/reconstruction/tecno/NoReconstructionCUDA.hpp"
#include "alsfvm/cuda/cuda_utils.hpp"

namespace alsfvm {
namespace reconstruction {
namespace tecno {

void NoReconstructionCUDA::performReconstruction(const volume::Volume&
    leftInput,
    const volume::Volume& rightInput,
    size_t,
    volume::Volume& leftOutput,
    volume::Volume& rightOutput) {
    for (size_t var = 0; var < leftInput.getNumberOfVariables(); ++var) {
        CUDA_SAFE_CALL(cudaMemcpyAsync(leftOutput.getScalarMemoryArea(
                    var)->getPointer(), leftInput.getScalarMemoryArea(var)->getPointer(),
                leftOutput.getScalarMemoryArea(var)->getSize() * sizeof(real),
                cudaMemcpyDeviceToDevice));
        CUDA_SAFE_CALL(cudaMemcpyAsync(rightOutput.getScalarMemoryArea(
                    var)->getPointer(), rightInput.getScalarMemoryArea(var)->getPointer(),
                rightOutput.getScalarMemoryArea(var)->getSize() * sizeof(real),
                cudaMemcpyDeviceToDevice));
    }
}

}
}
}
