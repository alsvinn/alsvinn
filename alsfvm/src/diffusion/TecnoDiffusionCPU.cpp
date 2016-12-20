#include "alsfvm/diffusion/TecnoDiffusionCPU.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/burgers/Burgers.hpp"
#include "alsfvm/diffusion/RoeMatrix.hpp"
#include "alsfvm/diffusion/RusanovMatrix.hpp"
#include "alsfvm/equation/equation_list.hpp"
#include <iostream>
#include <fstream>

namespace alsfvm { namespace diffusion { 
    namespace {
        template<class Equation, template<class, int> class DiffusionMatrix, int direction>
        void applyDiffusionCPU(Equation equation, volume::Volume& outputVolume, reconstruction::Reconstruction& reconstruction,
            volume::Volume& left, volume::Volume& right,
            volume::Volume& entropyVariables,
            const volume::Volume& conservedVolume) {
            volume::transform_volume<typename Equation::ConservedVariables,
                typename Equation::ConservedVariables >(conservedVolume, entropyVariables, [&](const typename Equation::ConservedVariables& in) {


                return equation.computeEntropyVariables(in);
            });

          
                for (size_t variable = 0; variable < outputVolume.getNumberOfVariables(); ++variable) {
                    volume::Volume variableVolumeEntropy(entropyVariables, { variable }, { "u" });
                    volume::Volume variableVolumeLeft(left, { variable }, { "u" });
                    volume::Volume variableVolumeRight(right, { variable }, { "u" });

                    reconstruction.performReconstruction(variableVolumeEntropy, direction, 0, variableVolumeLeft, variableVolumeRight);
                }
                //static int counter = 0;
                //counter++;
                //std::ofstream outFile("blah_" + std::to_string(counter) + ".txt");
                typename Equation::Views leftView(left);
                typename Equation::Views rightView(right);
                typename Equation::ConstViews conservedView(conservedVolume);
                typename Equation::Views outputView(outputVolume);
                volume::for_each_cell_index_with_neighbours(direction, left, [&](size_t leftIndex, size_t middleIndex, size_t rightIndex) {
                    auto diffusion = [&](size_t left, size_t right) {
                        auto leftValues = Equation::fetchConservedVariables(rightView, left);
                        auto rightValues = Equation::fetchConservedVariables(leftView, right);
                        

                        //auto conservedValues = Equation::fetchConservedVariables(conservedView, right);
                        auto conservedValues = 0.5*(Equation::fetchConservedVariables(conservedView, left) + Equation::fetchConservedVariables(conservedView, right));

                        DiffusionMatrix<Equation, direction> lambda(equation, conservedValues);

                        auto R = equation.template computeEigenVectorMatrix<direction>(conservedValues);
                        return 0.5*(R*(lambda*(R.transposed() * (rightValues - leftValues))));
                    };

                    //outFile << (diffusion(middleIndex, rightIndex) - diffusion(leftIndex, middleIndex))[1] << std::endl;
                    equation.addToViewAt(outputView, middleIndex, diffusion(middleIndex, rightIndex) - diffusion(leftIndex, middleIndex));
                }, make_direction_vector(direction), make_direction_vector(direction));
            
        }
    }
    template<class Equation, template<class, int> class DiffusionMatrix>
    TecnoDiffusionCPU<Equation, DiffusionMatrix>::TecnoDiffusionCPU(volume::VolumeFactory& volumeFactory,
        alsfvm::shared_ptr<reconstruction::Reconstruction> reconstruction,
        const simulator::SimulatorParameters& simulatorParameters)

        : 
        volumeFactory(volumeFactory), 
        reconstruction(reconstruction), 
        equation(static_cast<const typename Equation::Parameters&>(simulatorParameters.getEquationParameters()))
    {
        // empty
    }

    template<class Equation, template<class, int> class DiffusionMatrix>
    void TecnoDiffusionCPU<Equation, DiffusionMatrix>::applyDiffusion(volume::Volume& outputVolume,
        const volume::Volume& conservedVolume) 
    {



        if (!left || left->getNumberOfXCells() != conservedVolume.getNumberOfXCells()) {
            size_t nx = conservedVolume.getNumberOfXCells();
            size_t ny = conservedVolume.getNumberOfYCells();
            size_t nz = conservedVolume.getNumberOfZCells();

            size_t gcx = conservedVolume.getNumberOfXGhostCells();


            left = volumeFactory.createConservedVolume(nx, ny, nz, gcx);
            right = volumeFactory.createConservedVolume(nx, ny, nz, gcx);
            entropyVariables = volumeFactory.createConservedVolume(nx, ny, nz, gcx);

        }



        for (size_t direction = 0; direction < outputVolume.getDimensions(); ++direction) {
            if (direction == 0) {
                applyDiffusionCPU<Equation, DiffusionMatrix, 0>(equation, outputVolume, *reconstruction,
                    *left, *right,
                    *entropyVariables,
                    conservedVolume);
            }
            else if (direction == 1) {
                applyDiffusionCPU<Equation, DiffusionMatrix, 1>(equation, outputVolume, *reconstruction,
                    *left, *right,
                    *entropyVariables,
                    conservedVolume);
            }
            else if (direction == 2) {
                applyDiffusionCPU<Equation, DiffusionMatrix, 2>(equation, outputVolume, *reconstruction,
                    *left, *right,
                    *entropyVariables,
                    conservedVolume);
            }
        }
      
        
    }
    
    template<class Equation, template<class, int> class DiffusionMatrix>
    size_t TecnoDiffusionCPU<Equation, DiffusionMatrix>::getNumberOfGhostCells() const {
        return reconstruction->getNumberOfGhostCells();
    }

    template class TecnoDiffusionCPU<::alsfvm::equation::burgers::Burgers, ::alsfvm::diffusion::RoeMatrix>;
    template class TecnoDiffusionCPU<::alsfvm::equation::euler::Euler<1>, ::alsfvm::diffusion::RoeMatrix>;
    template class TecnoDiffusionCPU<::alsfvm::equation::euler::Euler<2>, ::alsfvm::diffusion::RoeMatrix>;
    template class TecnoDiffusionCPU<::alsfvm::equation::euler::Euler<3>, ::alsfvm::diffusion::RoeMatrix>;

    template class TecnoDiffusionCPU<::alsfvm::equation::burgers::Burgers, ::alsfvm::diffusion::RusanovMatrix>;
    template class TecnoDiffusionCPU<::alsfvm::equation::euler::Euler<1>,  ::alsfvm::diffusion::RusanovMatrix>;
    template class TecnoDiffusionCPU<::alsfvm::equation::euler::Euler<2>,  ::alsfvm::diffusion::RusanovMatrix>;
    template class TecnoDiffusionCPU<::alsfvm::equation::euler::Euler<3>,  ::alsfvm::diffusion::RusanovMatrix>;
}
}
