#include "alsfvm/diffusion/TecnoDiffusionCUDA.hpp"
#include "alsfvm/equation/burgers/Burgers.hpp"
#include "alsfvm/diffusion/RoeMatrix.hpp"
#include "alsfvm/equation/equation_list.hpp"
#include <iostream>
#include "alsfvm/cuda/cuda_utils.hpp"
#include "alsfvm/diffusion/RusanovMatrix.hpp"


namespace alsfvm {
namespace diffusion {
namespace {

template<class Equation, int direction>
__global__ void computeEntropyVariables(Equation equation,
    typename Equation::Views entropyLeftView,
    typename Equation::Views entropyRightView,
    const size_t numberOfXCells,
    const size_t numberOfYCells,
    const size_t numberOfZCells,
    typename Equation::ConstViews conservedView) {


    auto directionVector = make_direction_vector(direction);
    auto coordinates = cuda::getCoordinates(threadIdx, blockIdx, blockDim,
            numberOfXCells,
            numberOfYCells,
            numberOfZCells,
            directionVector);



    auto x = coordinates.x;
    auto y = coordinates.y;
    auto z = coordinates.z;

    if (x < 0 || y < 0 || z < 0) {
        return;
    }

    auto middleIndex = conservedView.index(x, y, z);
    auto leftIndex = conservedView.index(x - directionVector.x,
            y - directionVector.y,
            z - directionVector.z);




    auto conservedValuesMiddle = Equation::fetchConservedVariables(conservedView,
            middleIndex);
    auto conservedValuesLeft = Equation::fetchConservedVariables(conservedView,
            leftIndex);
    auto conservedValues = 0.5 * (conservedValuesMiddle + conservedValuesLeft);

    //
    auto RT = equation.template computeEigenVectorMatrix<direction>
        (conservedValues).transposed();


    equation.setViewAt(entropyLeftView, middleIndex,
        RT * equation.computeEntropyVariables(conservedValuesMiddle));
    equation.setViewAt(entropyRightView, leftIndex,
        RT * equation.computeEntropyVariables(conservedValuesLeft));

}

template<class Equation, template<typename, int> class DiffusionMatrix, int direction>
__global__ void multiplyDiffusionMatrix(Equation equation,
    typename Equation::Views output,
    typename Equation::ConstViews leftView,
    typename Equation::ConstViews rightView,
    typename Equation::ConstViews conservedView,
    int numberOfXCells,
    int numberOfYCells,
    int numberOfZCells,
    ivec3 directionVector
) {




    auto coordinates = cuda::getCoordinates(threadIdx, blockIdx, blockDim,
            numberOfXCells,
            numberOfYCells,
            numberOfZCells,
            directionVector);

    const int x = coordinates.x;
    const int y = coordinates.y;
    const int z = coordinates.z;



    if (x < 0 || y < 0 || z < 0) {
        return;
    }

    const size_t middleIndex = output.index(x, y, z);

    const size_t rightIndex = output.index(x + directionVector[0],
            y + directionVector[1],
            z + directionVector[2]);

    const size_t leftIndex = output.index(x - directionVector[0],
            y - directionVector[1],
            z - directionVector[2]);


    auto diffusion = [&](size_t left, size_t right) {
        auto leftValues = Equation::fetchConservedVariables(rightView, left);
        auto rightValues = Equation::fetchConservedVariables(leftView, right);

        //auto conservedValues = Equation::fetchConservedVariables(conservedView, right);

        auto conservedValuesMiddle = Equation::fetchConservedVariables(conservedView,
                rightIndex);
        auto conservedValuesLeft = Equation::fetchConservedVariables(conservedView,
                leftIndex);

        auto conservedValues = 0.5 * (conservedValuesMiddle + conservedValuesLeft);

        DiffusionMatrix<Equation, direction> matrix(equation, conservedValues);

        return -0.5 * ( equation.template computeEigenVectorMatrix<direction>
                (conservedValues) * (matrix * (leftValues - rightValues)));
    };


    equation.addToViewAt(output, middleIndex, diffusion(middleIndex,
            rightIndex) - diffusion(leftIndex, middleIndex));

}

template<class Equation, template<typename, int> class DiffusionMatrix, int direction>
void applyDiffusionCUDA(Equation equation, volume::Volume& outputVolume,
    reconstruction::tecno::TecnoReconstruction& reconstruction,
    volume::Volume& left, volume::Volume& right,
    volume::Volume& entropyVariablesLeft,
    volume::Volume& entropyVariablesRight,
    const volume::Volume& conservedVolume) {


    typename Equation::ConstViews conservedView(conservedVolume);
    typename Equation::Views entropyVariablesViewLeft(entropyVariablesLeft);
    typename Equation::Views entropyVariablesViewRight(entropyVariablesLeft);



    const size_t blockSize = 512;


    const ivec3 directionVector = make_direction_vector(direction);

    const ivec3 start = directionVector;
    const ivec3 end = conservedVolume.getTotalDimensions() - directionVector;




    auto launchParameters = cuda::makeKernelLaunchParameters(start, end, blockSize);

    auto gridSize = std::get<0>(launchParameters);
    auto numberOfCellsPerDimension = std::get<1>(launchParameters);
    CUDA_CHECK_IF_DEBUG;

    std::cout << gridSize << std::endl;
    std::cout << blockSize << std::endl;

    int minGridSize, blockSizeMax;
    CUDA_SAFE_CALL(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeMax,
            computeEntropyVariables<Equation, direction>));

    computeEntropyVariables<Equation, direction> << <gridSize, blockSize>> >
        (equation,
            entropyVariablesViewLeft,
            entropyVariablesViewRight,
            numberOfCellsPerDimension.x,
            numberOfCellsPerDimension.y,
            numberOfCellsPerDimension.z,
            conservedView);
    CUDA_CHECK_IF_DEBUG;




    reconstruction.performReconstruction(entropyVariablesLeft,
        entropyVariablesRight, direction, left, right);


    typename Equation::ConstViews leftView(left);
    typename Equation::ConstViews rightView(right);

    typename Equation::Views outputView(outputVolume);




    CUDA_CHECK_IF_DEBUG
    multiplyDiffusionMatrix<Equation, DiffusionMatrix, direction>
            << <gridSize, blockSize >> >(
                    equation, outputView,
                    leftView,
                    rightView,
                    conservedView,
                    numberOfCellsPerDimension.x,
                    numberOfCellsPerDimension.y,
                    numberOfCellsPerDimension.z,
                    directionVector
                );
    CUDA_CHECK_IF_DEBUG


}


}


template<class Equation, template<typename, int> class DiffusionMatrix>
TecnoDiffusionCUDA<Equation, DiffusionMatrix>::TecnoDiffusionCUDA(
    volume::VolumeFactory& volumeFactory,
    alsfvm::shared_ptr<reconstruction::tecno::TecnoReconstruction> reconstruction,
    const simulator::SimulatorParameters& simulatorParameters)

    :
    volumeFactory(volumeFactory),
    reconstruction(reconstruction),
    equation(static_cast<const typename Equation::Parameters&>
        (simulatorParameters.getEquationParameters())) {
    // empty
}

template<class Equation, template<typename, int> class DiffusionMatrix>
void TecnoDiffusionCUDA<Equation, DiffusionMatrix>::applyDiffusion(
    volume::Volume& outputVolume,
    const volume::Volume& conservedVolume) {
    if (!left || left->getNumberOfXCells() != conservedVolume.getNumberOfXCells()) {
        size_t nx = conservedVolume.getNumberOfXCells();
        size_t ny = conservedVolume.getNumberOfYCells();
        size_t nz = conservedVolume.getNumberOfZCells();

        size_t gcx = conservedVolume.getNumberOfXGhostCells();
        size_t gcy = conservedVolume.getNumberOfYGhostCells();
        size_t gcz = conservedVolume.getNumberOfZGhostCells();



        left = volumeFactory.createConservedVolume(nx, ny, nz, gcx);
        right = volumeFactory.createConservedVolume(nx, ny, nz, gcx);
        entropyVariablesLeft = volumeFactory.createConservedVolume(nx, ny, nz, gcx);
        entropyVariablesRight = volumeFactory.createConservedVolume(nx, ny, nz, gcx);

    }



    for (int direction = 0; direction < int(outputVolume.getDimensions());
        ++direction) {

        if (direction == 0) {
            applyDiffusionCUDA<Equation, DiffusionMatrix, 0>(equation, outputVolume,
                *reconstruction,
                *left, *right,
                *entropyVariablesLeft,
                *entropyVariablesRight,
                conservedVolume);
        } else if (direction == 1) {
            applyDiffusionCUDA<Equation, DiffusionMatrix, 1>(equation, outputVolume,
                *reconstruction,
                *left, *right,
                *entropyVariablesLeft,
                *entropyVariablesRight,
                conservedVolume);
        } else if (direction == 2) {
            applyDiffusionCUDA<Equation, DiffusionMatrix, 2>(equation, outputVolume,
                *reconstruction,
                *left, *right,
                *entropyVariablesLeft,
                *entropyVariablesRight,
                conservedVolume);
        }
    }
}

template<class Equation, template<typename, int> class DiffusionMatrix>
size_t TecnoDiffusionCUDA<Equation, DiffusionMatrix>::getNumberOfGhostCells()
const {
    return reconstruction->getNumberOfGhostCells();
}

template class
TecnoDiffusionCUDA<::alsfvm::equation::burgers::Burgers, ::alsfvm::diffusion::RoeMatrix >;
template class
TecnoDiffusionCUDA<::alsfvm::equation::euler::Euler<1>, ::alsfvm::diffusion::RoeMatrix >;
template class
TecnoDiffusionCUDA<::alsfvm::equation::euler::Euler<2>, ::alsfvm::diffusion::RoeMatrix >;
template class
TecnoDiffusionCUDA<::alsfvm::equation::euler::Euler<3>, ::alsfvm::diffusion::RoeMatrix >;

template class
TecnoDiffusionCUDA<::alsfvm::equation::burgers::Burgers, ::alsfvm::diffusion::RusanovMatrix >;
template class
TecnoDiffusionCUDA<::alsfvm::equation::euler::Euler<1>,  ::alsfvm::diffusion::RusanovMatrix >;
template class
TecnoDiffusionCUDA<::alsfvm::equation::euler::Euler<2>,  ::alsfvm::diffusion::RusanovMatrix >;
template class
TecnoDiffusionCUDA<::alsfvm::equation::euler::Euler<3>,  ::alsfvm::diffusion::RusanovMatrix >;
}
}
