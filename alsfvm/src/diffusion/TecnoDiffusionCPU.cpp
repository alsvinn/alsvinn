#include "alsfvm/diffusion/TecnoDiffusionCPU.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/burgers/Burgers.hpp"
#include "alsfvm/diffusion/RoeMatrix.hpp"

namespace alsfvm { namespace diffusion { 

    template<class Equation, class DiffusionMatrix>
    TecnoDiffusionCPU<Equation, DiffusionMatrix>::TecnoDiffusionCPU(volume::VolumeFactory& volumeFactory,
        alsfvm::shared_ptr<reconstruction::Reconstruction>& reconstruction,
        const alsfvm::shared_ptr<simulator::SimulatorParameters>& simulatorParameters,
        size_t nx, size_t ny, size_t nz)

        : 
        reconstruction(reconstruction), equation(static_cast<typename Equation::Parameters&>(simulatorParameters->getEquationParameters()))
    {
        
        left = volumeFactory.createConservedVolume(nx, ny, nz);
        right = volumeFactory.createConservedVolume(nx, ny, nz);
        entropyVariables = volumeFactory.createConservedVolume(nx, ny, nz);

    }

    template<class Equation, class DiffusionMatrix>
    void TecnoDiffusionCPU<Equation, DiffusionMatrix>::applyDiffusion(volume::Volume& outputVolume,
        const volume::Volume& conservedVolume) 
    {
        volume::transform_volume<typename Equation::ConservedVariables,
            typename Equation::ConservedVariables >(conservedVolume, *entropyVariables, [&](const typename Equation::ConservedVariables& in) {
            // R^T * v
            return (equation.computeEigenVectorMatrix(in).transposed())*equation.computeEntropyVariables(in);
        });

        for (int direction = 0; direction < outputVolume.getDimensions(); ++direction) {
            reconstruction->performReconstruction(*entropyVariables, direction, 0, *left, *right);

            typename Equation::Views leftView(*left);
            typename Equation::Views rightView(*right);
            typename Equation::ConstViews conservedView(conservedVolume);
            typename Equation::Views outputView(outputVolume);

            volume::for_each_cell_index(conservedVolume, [&](size_t index) {

                auto leftValues = Equation::fetchConservedVariables(leftView, index);
                auto rightValues = Equation::fetchConservedVariables(rightView, index);

                auto conservedValues = Equation::fetchConservedVariables(conservedView, index);

                DiffusionMatrix matrix(equation, conservedValues);

                auto output = equation.computeEigenVectorMatrix(conservedValues) * (matrix * (leftValues - rightValues));

                Equation::addToViewAt(outputView, index, output);
            });
        }
    }

    template TecnoDiffusionCPU<equation::burgers::Burgers, RoeMatrix<equation::burgers::Burgers> >;
}
}