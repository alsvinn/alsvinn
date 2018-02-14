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
