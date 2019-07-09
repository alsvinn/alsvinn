#pragma once
#include <thrust/device_vector.h>
#include "alsfvm/functional/structure_common.hpp"
namespace alsfvm {

namespace functional {



//! Computes the structure function for FIXED h
//!
//! The goal is to compute the structure function, then reduce (sum) over space
//! then go on to next h
//!
template<class PowerClass>
__global__ void computeStructureCubeKernel(real* output,
    alsfvm::memory::View<const real> input,
    int h,
    int nx, int ny, int nz, int ngx, int ngy, int ngz,
    real p, int dimensions) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index >= nx * ny * nz) {
        return;
    }

    const int i = index % nx;
    const int j = (index / nx) % ny;
    const int k = (index / nx / ny);


    output[index] = 0;
    forEachPointInComputeStructureCube([&] (double u, double u_h) {
        output[index] += PowerClass::power(fabs(u - u_h), p);
    }, input, i, j, k, h, nx, ny, nz, ngx, ngy, ngz, dimensions);

}


template<class PowerClass, class BufferClass>
void computeStructureCubeCUDA(alsfvm::volume::Volume& output,
    const alsfvm::volume::Volume& input,
    BufferClass& buffer,

    size_t numberOfH, double p) {
    for (size_t var = 0; var < input.getNumberOfVariables(); ++var) {
        auto inputView = input[var]->getView();
        auto outputView = output[var]->getView();

        const int ngx = int(input.getNumberOfXGhostCells());
        const int ngy = int(input.getNumberOfYGhostCells());
        const int ngz = int(input.getNumberOfZGhostCells());

        const int nx = int(input.getNumberOfXCells()) - 2 * ngx;
        const int ny = int(input.getNumberOfYCells()) - 2 * ngy;
        const int nz = int(input.getNumberOfZCells()) - 2 * ngz;

        buffer.resize(nx * ny * nz);
        const int dimensions = input.getDimensions();

        for (int h = 1; h < numberOfH; ++h) {
            const int threads = 1024;
            const int size = nx * ny * nz;
            const int blockNumber = (size + threads - 1) / threads;


            computeStructureCubeKernel<PowerClass> <<< blockNumber, threads>>>
            (thrust::raw_pointer_cast(
                    buffer.data()),
                inputView,
                h, nx, ny, nz, ngx, ngy, ngz, p, dimensions);


            real structureResult = thrust::reduce(buffer.begin(),
                    buffer.end(),
                    0.0, thrust::plus<real>());

            outputView.at(h) += structureResult / (nx * ny * nz);
        }

    }
}


inline void dispatchComputeStructureCubeCUDA(alsfvm::volume::Volume& output,
    const alsfvm::volume::Volume& input, thrust::device_vector<real>& buffer,
    int numberOfH, double p) {
    if (p == 1.0) {
        computeStructureCubeCUDA < alsutils::math::FastPower<1>>
            (output, input, buffer, numberOfH, p);
    } else if (p == 2.0) {
        computeStructureCubeCUDA < alsutils::math::FastPower<2>>
            (output, input, buffer, numberOfH, p);
    }

    else if (p == 3.0) {
        computeStructureCubeCUDA < alsutils::math::FastPower<3>>
            (output, input, buffer, numberOfH, p);
    } else if (p == 4.0) {
        computeStructureCubeCUDA < alsutils::math::FastPower<4>>
            (output, input, buffer, numberOfH, p);
    } else if (p == 5.0) {
        computeStructureCubeCUDA < alsutils::math::FastPower<5>>
            (output, input, buffer, numberOfH, p);
    } else {
        computeStructureCubeCUDA < alsutils::math::PowPower>
        (output, input, buffer, numberOfH, p);
    }
}
}

}
