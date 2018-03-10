#pragma once
#include <cuda.h>

#include "alsfvm/types.hpp"
namespace alsfvm {
namespace cuda {

dim3 makeBlockDimension(const ivec3& size, const int maximumBlockSize = 1024) {
    // We prioritize the x direction
    dim3 blockDimension;

    // We don't want to allocate too much
    if (size.x < maximumBlockSize) {
        blockDimension.x = size.x;

    } else if (size.x % maximumBlockSize == 0) {
        blockDimension.x = maximumBlockSize;
    } else {
        blockDimension.x = size.x;

        // Here we take (we hope, obviously doesn't work if blockDimension.x is a prime)
        // to take the nearest denominator of size.x
        while (blockDimension.x > maximumBlockSize) {
            blockDimension.x /= 2;
        }
    }

    blockDimension.y = 1;
    blockDimension.z = 1;

    return blockDimension;
}

dim3 makeGridDimension(const ivec3& size, const int maximumBlockSize = 1024) {
    dim3 gridDimension;
    dim3 blockSize = makeBlockDimension(size, maximumBlockSize);

    gridDimension.x = (size.x + blockSize.x - 1) / blockSize.x;
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
