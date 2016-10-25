#include "alsfvm/diffusion/TecnoDiffusionCPU.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/burgers/Burgers.hpp"
#include "alsfvm/diffusion/RoeMatrix.hpp"
#include <iostream>

namespace alsfvm { namespace diffusion { 

    template<class Equation, class DiffusionMatrix>
    TecnoDiffusionCPU<Equation, DiffusionMatrix>::TecnoDiffusionCPU(volume::VolumeFactory& volumeFactory,
        alsfvm::shared_ptr<reconstruction::Reconstruction>& reconstruction,
        const simulator::SimulatorParameters& simulatorParameters)

        : 
        volumeFactory(volumeFactory), 
        reconstruction(reconstruction), 
        equation(static_cast<const typename Equation::Parameters&>(simulatorParameters.getEquationParameters()))
    {
        // empty
    }

    template<class Equation, class DiffusionMatrix>
    void TecnoDiffusionCPU<Equation, DiffusionMatrix>::applyDiffusion(volume::Volume& outputVolume,
        const volume::Volume& conservedVolume) 
    {
        if (!left || left->getNumberOfXCells() != conservedVolume.getNumberOfXCells()) {
            size_t nx = conservedVolume.getNumberOfXCells();
            size_t ny = conservedVolume.getNumberOfYCells();
            size_t nz = conservedVolume.getNumberOfZCells();

            size_t gcx = conservedVolume.getNumberOfXGhostCells();
            size_t gcy = conservedVolume.getNumberOfYGhostCells();
            size_t gcz = conservedVolume.getNumberOfZGhostCells();



            left = volumeFactory.createConservedVolume(nx, ny, nz, gcx);
            right = volumeFactory.createConservedVolume(nx, ny, nz, gcx);
            entropyVariables = volumeFactory.createConservedVolume(nx, ny, nz, gcx);

        }
        volume::transform_volume<typename Equation::ConservedVariables,
            typename Equation::ConservedVariables >(conservedVolume, *entropyVariables, [&](const typename Equation::ConservedVariables& in) {

           
            return equation.computeEntropyVariablesMultipliedByEigenVectorMatrix(in);
        });

        for (int direction = 0; direction < outputVolume.getDimensions(); ++direction) {
            reconstruction->performReconstruction(*entropyVariables, direction, 0, *left, *right);

            typename Equation::Views leftView(*left);
            typename Equation::Views rightView(*right);
            typename Equation::ConstViews conservedView(conservedVolume);
            typename Equation::Views outputView(outputVolume);

            volume::for_each_cell_index_with_neighbours(direction, *left, [&](size_t leftIndex, size_t middleIndex, size_t rightIndex) {
                auto diffusion = [&](size_t left, size_t right) {
                    auto leftValues = Equation::fetchConservedVariables(rightView, left);
                    auto rightValues = Equation::fetchConservedVariables(leftView, right);

                    auto conservedValues = Equation::fetchConservedVariables(conservedView, right);

                    DiffusionMatrix matrix(equation, conservedValues);

                    return 0.5*(equation.computeEigenVectorMatrix(conservedValues) * (matrix * (leftValues - rightValues)));
                };


                Equation::addToViewAt(outputView, middleIndex, diffusion(middleIndex, rightIndex)-diffusion(leftIndex, middleIndex));
            });
        }
    }
    
    template<class Equation, class DiffusionMatrix>
    size_t TecnoDiffusionCPU<Equation, DiffusionMatrix>::getNumberOfGhostCells() const {
        return reconstruction->getNumberOfGhostCells();
    }

    template TecnoDiffusionCPU<equation::burgers::Burgers, RoeMatrix<equation::burgers::Burgers> >;
}
}