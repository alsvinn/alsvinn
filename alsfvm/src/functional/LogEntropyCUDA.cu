#include "alsfvm/functional/LogEntropyCUDA.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/functional/register_functional.hpp"
#include <iostream>
#include "alsfvm/gpu_array.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsutils/cuda/cuda_safe_call.hpp"
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>

namespace alsfvm {
namespace functional {

namespace {

template<int numberOfDimensions>
__global__ void logEntropyKernel(memory::View<real> output,
                               memory::View<const real> densityView,
                                 memory::View<const real> energyView,


                                gpu_array<const real*, numberOfDimensions> momentumPointers,
                               int ngx, int ngy, int ngz,
                               real dxdydz, real gamma)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if ( index >= output.size()) {
        return;
    }

    const size_t x = index % output.getNumberOfXCells();
    const size_t y = (index / output.getNumberOfXCells()) % output.getNumberOfYCells();
    const size_t z = index / (output.getNumberOfXCells()*output.getNumberOfYCells());
    const real density = densityView.at(x + ngx, y + ngy, z + ngz);
    const real energy = energyView.at(x + ngx, y + ngy, z + ngz);

    const auto indexOfInput = energyView.index(x + ngx, y + ngy, z + ngz);

    real momentumSquared = 0.0;

    for (size_t component = 0; component < numberOfDimensions; ++component) {
        const real momentum = momentumPointers[component][indexOfInput];
        momentumSquared += momentum*momentum;
    }

    const real pressure = (gamma - 1) * (energy - 1.0 / (2 * density) *
            momentumSquared);


    const real s = log(pressure) - gamma * log(density);
    const real E = (-density * s) / (gamma - 1);
    output.at(index) = E * dxdydz;
}

template<int numberOfDimensions>
void runLogEntropyKernel(memory::View<real> output,
                         memory::View<const real> densityView,
                           memory::View<const real> energyView,


                          std::vector<const real*> momentumPointers,
                          ivec3 ghostCells,
                         real dxdydz, real gamma) {

    const size_t threads = 1024;
    const size_t size = output.size();
    gpu_array<const real*, numberOfDimensions> momentumPointersForGPU;
     for (size_t component = 0; component < numberOfDimensions; ++component) {
         momentumPointersForGPU[component] = momentumPointers[component];
     }

    logEntropyKernel<numberOfDimensions><<<(size + threads -1)/threads, threads>>>(output, densityView,
                                                                       energyView,
                                                                       momentumPointersForGPU,
                                                            ghostCells.x,
                                                            ghostCells.y,
                                                            ghostCells.z,
                                                            dxdydz,
                                                                       gamma);

}

void runLogEntropyKernel(memory::View<real> output,
                         memory::View<const real> densityView,
                           memory::View<const real> energyView,


                          std::vector<const real*> momentumPointers,
                          ivec3 ghostCells,
                         real dxdydz, real gamma, size_t numberOfDimensions) {
    if (numberOfDimensions == 1) {
        runLogEntropyKernel<1>(output, densityView, energyView, momentumPointers, ghostCells, dxdydz, gamma);
    } else if (numberOfDimensions ==2) {
        runLogEntropyKernel<2>(output, densityView, energyView, momentumPointers, ghostCells, dxdydz, gamma);
    } else if (numberOfDimensions == 3) {
        runLogEntropyKernel<3>(output, densityView, energyView, momentumPointers, ghostCells, dxdydz, gamma);
    } else {
        THROW("Unsupported number of dimensions " << numberOfDimensions);
    }
}
}

LogEntropyCUDA::LogEntropyCUDA(const Functional::Parameters& parameters)
    : gamma(parameters.getDouble("gamma")) {


}

void LogEntropyCUDA::operator()(volume::Volume& conservedVolumeOut,
    const volume::Volume& conservedVolumeIn, const real weight,
    const grid::Grid& grid) {

    const auto lengths = grid.getCellLengths();

    const real dxdydz = lengths.x * lengths.y * lengths.z;

    const auto& densityView = conservedVolumeIn.getScalarMemoryArea(
            "rho")->getView();

    const auto& energyView = conservedVolumeIn.getScalarMemoryArea(
            "E")->getView();

    const size_t numberOfComponents = grid.getActiveDimension();

    std::vector<const real*> momentumPointers;


    momentumPointers.push_back(
        conservedVolumeIn.getScalarMemoryArea("mx")->getPointer());

    if (numberOfComponents > 1) {
        momentumPointers.push_back(
            conservedVolumeIn.getScalarMemoryArea("my")->getPointer());
    }

    if (numberOfComponents > 2) {
        momentumPointers.push_back(
            conservedVolumeIn.getScalarMemoryArea("mz")->getPointer());
    }

    const auto nx = conservedVolumeIn.getNumberOfXCells();
    const auto ny = conservedVolumeIn.getNumberOfYCells();
    const auto nz = conservedVolumeIn.getNumberOfZCells();
    const auto totalSize = nx*ny*nz;
    if (!buffer) {

        buffer = alsfvm::make_shared<cuda::CudaMemory<real>>(nx, ny, nz);
        buffer->makeZero();
        bufferOut = alsfvm::make_shared<cuda::CudaMemory<real>>(1);
    }

    bufferOut->makeZero();
    auto outputView = buffer->getView();

    const auto ghostCells = conservedVolumeIn.getNumberOfGhostCells();

    runLogEntropyKernel(outputView, densityView, energyView, momentumPointers, ghostCells, dxdydz, gamma, numberOfComponents);

    if (!temporaryReductionMemory) {

        temporaryReductionMemoryStorageSizeBytes = 0;
        void* rawPointerStorage = nullptr;

        // This is a bit weird, but the first time we call it with a null pointer,
        // it will tell us the size we need to allocate. Again, read the documentation
        // at http://nvlabs.github.io/cub/example_device_reduce_8cu-example.html
        CUDA_SAFE_CALL(cub::DeviceReduce::Sum(rawPointerStorage,
                                         temporaryReductionMemoryStorageSizeBytes,
                                         buffer->getPointer(),
                                         bufferOut->getPointer(),
                                         totalSize));

        // we get size in bytes, but we want to allocate whole reals (makes setup easier,
        // no real technical reason), we just make sure we allocate at least temporaryReductionMemoryStorageSizeBytes bytes:
        temporaryReductionMemory.reset(new cuda::CudaMemory<real>((temporaryReductionMemoryStorageSizeBytes+sizeof(real)-1)/sizeof(real)));

    }

    CUDA_SAFE_CALL(cub::DeviceReduce::Sum(temporaryReductionMemory->getPointer(),
                                     temporaryReductionMemoryStorageSizeBytes,
                                     buffer->getPointer(),
                                     bufferOut->getPointer(),
                                     totalSize));

    *bufferOut *= weight;

    (*conservedVolumeOut.getScalarMemoryArea("E")) += *bufferOut;

}

ivec3 LogEntropyCUDA::getFunctionalSize(const grid::Grid&) const {
    return {1, 1, 1};
}
REGISTER_FUNCTIONAL(cuda, log_entropy, LogEntropyCUDA)
}
}
