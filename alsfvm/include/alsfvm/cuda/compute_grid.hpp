#pragma once
#include <cuda.h>

#include "alsfvm/types.hpp"
namespace alsfvm {
namespace cuda {

dim3 makeBlockDimension(const ivec3& size, const int maximumBlockSize = 1024) {
    // We prioritize the x direction
    dim3 blockDimension;
    blockDimension.x = maximumBlockSize;
    blockDimension.y = 1;
    blockDimension.z = 1;

    return blockDimension;
}

dim3 makeGridDimension(const ivec3& size, const int maximumBlockSize = 1024) {
    dim3 gridDimension;
    gridDimension.x = (size.x + maximumBlockSize - 1) / maximumBlockSize;
    gridDimension.y = size.y;
    gridDimension.z = size.z;

    return gridDimension;
}

__device__ ivec3 getInternalFormat(const dim3& thread, const dim3& block,
    const dim3& blockDim) {

    return ivec3{thread.x + blockDim.x * block.x,
            thread.y + blockDim.y * block.y,
            thread.z + blockDim.z * block.z};
}
}
}
