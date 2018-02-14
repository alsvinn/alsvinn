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
