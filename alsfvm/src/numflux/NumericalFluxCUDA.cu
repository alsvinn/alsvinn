#include "alsfvm/numflux/NumericalFluxCUDA.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"
#include "alsfvm/numflux/euler/HLL3.hpp"
#include <thrust/device_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "alsfvm/numflux/numflux_util.hpp"
#include <iostream>
#include "alsfvm/numflux/numerical_flux_list.hpp"
#include "alsfvm/cuda/cuda_utils.hpp"
#include "alsfvm/cuda/compute_grid.hpp"

namespace alsfvm {
namespace numflux {

namespace {

template<class Equation, size_t dimension, bool xDir, bool yDir, bool zDir, size_t direction>
__global__ void combineFluxDevice(Equation equation,
    typename Equation::ConstViews input, typename Equation::Views output,
    const size_t numberOfXCells, const size_t numberOfYCells,
    const size_t numberOfZCells, const size_t numberOfGhostCells,
    ivec3 start, ivec3 end) {

    auto internalFormat = cuda::getInternalFormat(threadIdx, blockIdx, blockDim);
    const size_t xInternalFormat = internalFormat.x;
    const size_t yInternalFormat = internalFormat.y;
    const size_t zInternalFormat = internalFormat.z;

    //printf(" %d %d %d\n", xInternalFormat, yInternalFormat, zInternalFormat);
    if (xInternalFormat >= numberOfXCells || yInternalFormat >= numberOfYCells
        || zInternalFormat >= numberOfZCells) {
        return;
    }

    const size_t x = xInternalFormat + numberOfGhostCells - xDir + start.x;
    const size_t y = yInternalFormat + (dimension > 1) * numberOfGhostCells - yDir +
        start.y;
    const size_t z = zInternalFormat + (dimension > 2) * numberOfGhostCells - zDir +
        start.z;



    const size_t rightIndex = output.index(x + xDir, y + yDir, z + zDir);
    const size_t middleIndex = output.index(x, y, z);

    typename Equation::ConservedVariables fluxMiddle = (-1.0) *
        equation.fetchConservedVariables(input, middleIndex);
    typename Equation::ConservedVariables fluxRight =
        equation.fetchConservedVariables(input, rightIndex);
    equation.addToViewAt(output, rightIndex, (fluxMiddle + fluxRight));
}



template<class Flux, class Equation, size_t dimension, bool xDir, bool yDir, bool zDir, size_t direction>
__global__ void computeFluxDevice(Equation equation,
    typename Equation::ConstViews left,
    typename Equation::ConstViews right,
    typename Equation::Views output,
    const size_t numberOfXCells,
    const size_t numberOfYCells,
    const size_t numberOfZCells,
    real* waveSpeeds,
    const size_t numberOfGhostCells, ivec3 start, ivec3 end) {

    auto internalFormat = cuda::getInternalFormat(threadIdx, blockIdx, blockDim);
    const size_t xInternalFormat = internalFormat.x;
    const size_t yInternalFormat = internalFormat.y;
    const size_t zInternalFormat = internalFormat.z;
    if (xInternalFormat >= numberOfXCells || yInternalFormat >= numberOfYCells
        || zInternalFormat >= numberOfZCells) {
        return;
    }

    const size_t index = xInternalFormat
            + yInternalFormat * numberOfXCells
            + zInternalFormat * numberOfXCells * numberOfYCells;
    const size_t x = xInternalFormat +  numberOfGhostCells - xDir + start.x;
    const size_t y = yInternalFormat + (dimension > 1) * (numberOfGhostCells - yDir)
        + start.y;
    const size_t z = zInternalFormat + (dimension > 2) * (numberOfGhostCells - zDir)
        + start.z;




    auto stencil = getStencil<Flux>(Flux());
    // Now we need to build up the stencil for this set of indices
    decltype(stencil) indices;

    for (int stencilIndex = 0; stencilIndex < stencil.size(); ++stencilIndex) {
        indices[stencilIndex] = output.index(
                x + xDir * stencil[stencilIndex],
                y + yDir * stencil[stencilIndex],
                z + zDir * stencil[stencilIndex]);
    }




    typename Equation::ConservedVariables fluxMiddleRight;
    auto outputIndex = output.index(x, y, z);
    waveSpeeds[index] = computeFluxForStencil<Flux, Equation, direction> (equation,
            indices, left, right, fluxMiddleRight);



    equation.setViewAt(output, outputIndex, (-1.0)*fluxMiddleRight);
}

template<class Equation, size_t dimension>
__global__ void makeZeroDevice(Equation equation,
    typename Equation::Views view,
    const size_t numberOfXCells,
    const size_t numberOfYCells,
    const size_t numberOfZCells,

    const size_t numberOfGhostCells, ivec3 start, ivec3 end) {
    auto internalFormat = cuda::getInternalFormat(threadIdx, blockIdx, blockDim);
    const size_t xInternalFormat = internalFormat.x;
    const size_t yInternalFormat = internalFormat.y;
    const size_t zInternalFormat = internalFormat.z;
    if (xInternalFormat >= numberOfXCells || yInternalFormat >= numberOfYCells
        || zInternalFormat >= numberOfZCells) {
        return;
    }

    const size_t x = xInternalFormat + numberOfGhostCells  + start.x;
    const size_t y = yInternalFormat + (dimension > 1) * numberOfGhostCells  +
        start.y;
    const size_t z = zInternalFormat + (dimension > 2) * numberOfGhostCells  +
        start.z;

    typename Equation::ConservedVariables zero;
    equation.setViewAt(view, view.index(x, y, z), zero);

}

template< class Equation, size_t dimension>
void makeZero(volume::Volume& volume,
    const ivec3& start,
    const ivec3& end,
    const Equation& equation) {
    const int numberOfGhostCells = volume.getNumberOfGhostCells().x;
    const int numberOfXCells = int(volume.getTotalNumberOfXCells()) - 2 *
        numberOfGhostCells - start.x + end.x;
    const int numberOfYCells = int(volume.getTotalNumberOfYCells()) - 2 *
        (dimension > 1) * numberOfGhostCells  - start.y + end.y;
    const int numberOfZCells = int(volume.getTotalNumberOfZCells()) - 2 *
        (dimension > 2) * numberOfGhostCells  - start.z + end.z;



    typename Equation::Views viewOut(volume);

    size_t totalSize = numberOfXCells * numberOfYCells * numberOfZCells;
    size_t blockSize = 128;

    auto blockDim = cuda::makeBlockDimension({numberOfXCells,
                                             numberOfYCells,
                                             numberOfZCells},
                                             blockSize);

    auto gridDim = cuda::makeGridDimension({numberOfXCells,
                                             numberOfYCells,
                                             numberOfZCells},
                                             blockSize);

    makeZeroDevice<Equation, dimension>
            << <gridDim, blockDim>> >
                (equation, viewOut, numberOfXCells, numberOfYCells, numberOfZCells,
                    numberOfGhostCells, start, end);

}

template<class Flux, class Equation,  size_t dimension, bool xDir, bool yDir, bool zDir, size_t direction>
void computeFlux(const Equation& equation,
    const volume::Volume& left,
    const volume::Volume& right,
    volume::Volume& output,
    int numberOfGhostCells,
    real& waveSpeed,  const ivec3& start,
    const ivec3& end) {
    static thrust::device_vector<real> waveSpeeds;
    waveSpeeds.resize(left.getScalarMemoryArea(0)->getSize(), 0.0);

    const int numberOfXCells = int(left.getTotalNumberOfXCells()) - 2 *
        numberOfGhostCells + xDir - start.x + end.x;
    const int numberOfYCells = int(left.getTotalNumberOfYCells()) - 2 *
        (dimension > 1) * numberOfGhostCells + yDir - start.y + end.y;
    const int numberOfZCells = int(left.getTotalNumberOfZCells()) - 2 *
        (dimension > 2) * numberOfGhostCells + zDir - start.z + end.z;
    typename Equation::ConstViews viewLeft(left);
    typename Equation::ConstViews viewRight(right);
    typename Equation::Views viewOut(output);

    size_t totalSize = numberOfXCells * numberOfYCells * numberOfZCells;
    const size_t blockSize = 128;
    auto blockDim = cuda::makeBlockDimension({numberOfXCells,
                                             numberOfYCells,
                                             numberOfZCells},
                                             blockSize);

    auto gridDim = cuda::makeGridDimension({numberOfXCells,
                                             numberOfYCells,
                                             numberOfZCells},
                                             blockSize);


    computeFluxDevice <Flux, Equation, dimension, xDir, yDir, zDir, direction>
            << < gridDim, blockDim >> >
                (equation, viewLeft, viewRight, viewOut, numberOfXCells,
                    numberOfYCells, numberOfZCells,
                    thrust::raw_pointer_cast(&waveSpeeds[0]), numberOfGhostCells, start, end);

    waveSpeed = thrust::reduce(waveSpeeds.begin(), waveSpeeds.begin() + totalSize,
            0.0, thrust::maximum<real>());

}


template<class Equation, size_t dimension, bool xDir, bool yDir, bool zDir, size_t direction>
void combineFlux(const Equation& equation, const volume::Volume& input,
    volume::Volume& output, int numberOfGhostCells
    ,  const ivec3& start,
    const ivec3& end) {

    const int numberOfXCells = int(input.getTotalNumberOfXCells()) - 2 *
        numberOfGhostCells - start.x + end.x;
    const int numberOfYCells = int(input.getTotalNumberOfYCells()) - 2 *
        (dimension > 1) * numberOfGhostCells  - start.y + end.y;
    const int numberOfZCells = int(input.getTotalNumberOfZCells()) - 2 *
        (dimension > 2) * numberOfGhostCells  - start.z + end.z;

    typename Equation::ConstViews inputView(input);
    typename Equation::Views viewOut(output);

    size_t totalSize = numberOfXCells * numberOfYCells * numberOfZCells;
    const size_t blockSize = 128;
    auto blockDim = cuda::makeBlockDimension({numberOfXCells,
                                             numberOfYCells,
                                             numberOfZCells},
                                             blockSize);

    auto gridDim = cuda::makeGridDimension({numberOfXCells,
                                             numberOfYCells,
                                             numberOfZCells},
                                             blockSize);


    combineFluxDevice <Equation, dimension, xDir, yDir, zDir, direction>
            << < gridDim, blockDim >> >
                (equation, inputView, viewOut, numberOfXCells, numberOfYCells, numberOfZCells,
                    numberOfGhostCells, start, end);

}

template<class Flux, class Equation, size_t dimension>
void callComputeFlux(const Equation& equation,
    const volume::Volume& conservedVariables, volume::Volume& left,
    volume::Volume& right, volume::Volume& output, volume::Volume& temporaryOutput,
    size_t numberOfGhostCells, rvec3& waveSpeeds,
    reconstruction::Reconstruction& reconstruction,  const ivec3& start,
    const ivec3& end) {
    reconstruction.performReconstruction(conservedVariables, 0, 0, left, right,
        start, end);
    computeFlux<Flux, Equation, dimension, 1, 0, 0, 0>(equation, left, right,
        temporaryOutput,
        numberOfGhostCells,
        waveSpeeds.x,
        start, end);
    combineFlux<Equation, dimension, 1, 0, 0, 0>(equation,
        temporaryOutput,
        output,
        numberOfGhostCells,
        start,
        end);

    if (dimension > 1) {
        reconstruction.performReconstruction(conservedVariables, 1, 0, left, right,
            start, end);
        computeFlux<Flux, Equation, dimension, 0, 1, 0, 1>(equation, left, right,
            temporaryOutput, numberOfGhostCells, waveSpeeds.y, start, end);
        combineFlux<Equation, dimension, 0, 1, 0, 1>(equation, temporaryOutput, output,
            numberOfGhostCells, start, end);
    }

    if (dimension > 2) {
        reconstruction.performReconstruction(conservedVariables, 2, 0, left, right,
            start, end);
        computeFlux<Flux, Equation, dimension, 0, 0, 1, 2>(equation, left, right,
            temporaryOutput, numberOfGhostCells, waveSpeeds.z, start, end);
        combineFlux<Equation, dimension, 0, 0, 1, 2>(equation, temporaryOutput, output,
            numberOfGhostCells, start, end);
    }

}

}

template<class Flux, class Equation, size_t dimension>
NumericalFluxCUDA<Flux, Equation, dimension>::NumericalFluxCUDA(
    const grid::Grid& grid,
    alsfvm::shared_ptr<reconstruction::Reconstruction>& reconstruction,
    const simulator::SimulatorParameters& parameters,
    alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration)
    : reconstruction(reconstruction),
      equationParameters(static_cast<const typename Equation::Parameters&>
          (parameters.getEquationParameters())),
      equation(equationParameters) {
    static_assert(dimension > 0, "We only support positive dimension!");
    static_assert(dimension < 4, "We only support dimension up to 3");

    alsfvm::shared_ptr<memory::MemoryFactory> memoryFactory(
        new memory::MemoryFactory(deviceConfiguration));
    volume::VolumeFactory volumeFactory(Equation::name, memoryFactory);

    left = volumeFactory.createConservedVolume(grid.getDimensions().x,
            grid.getDimensions().y,
            grid.getDimensions().z,
            getNumberOfGhostCells());
    left->makeZero();

    right = volumeFactory.createConservedVolume(grid.getDimensions().x,
            grid.getDimensions().y,
            grid.getDimensions().z,
            getNumberOfGhostCells());

    right->makeZero();

    fluxOutput = volumeFactory.createConservedVolume(grid.getDimensions().x,
            grid.getDimensions().y,
            grid.getDimensions().z,
            getNumberOfGhostCells());
}

template<class Flux, class Equation, size_t dimension>
void NumericalFluxCUDA<Flux, Equation, dimension>::computeFlux(
    const volume::Volume& conservedVariables,
    rvec3& waveSpeeds, bool computeWaveSpeeds,
    volume::Volume& output, const ivec3& start,
    const ivec3& end     ) {

    static_assert(dimension > 0, "We only support positive dimension!");
    static_assert(dimension < 4, "We only support dimension up to 3");

    makeZero<Equation, dimension>(output, start,
        end, equation);

    callComputeFlux<Flux, Equation, dimension>(equation,
        conservedVariables,
        *left,
        *right,
        output,
        *fluxOutput,
        getNumberOfGhostCells(),
        waveSpeeds,
        *reconstruction,
        start,
        end);
}

///
/// \returns the number of ghost cells this specific flux requires
///
template<class Flux, class Equation, size_t dimension>
size_t NumericalFluxCUDA<Flux, Equation, dimension>::getNumberOfGhostCells() {
    return max(getStencil<Flux>(Flux()).size() - 1,
            reconstruction->getNumberOfGhostCells());
}

ALSFVM_FLUX_INSTANTIATE(NumericalFluxCUDA)
}
}
