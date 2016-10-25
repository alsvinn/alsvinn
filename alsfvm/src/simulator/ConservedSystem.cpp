#include "alsfvm/simulator/ConservedSystem.hpp"
#include <algorithm>
namespace alsfvm { namespace simulator {

ConservedSystem::ConservedSystem(alsfvm::shared_ptr<numflux::NumericalFlux> numericalFlux,
    alsfvm::shared_ptr<diffusion::DiffusionOperator> diffusionOperator)
    : numericalFlux(numericalFlux), diffusionOperator(diffusionOperator)
{

}

void ConservedSystem::operator()(const volume::Volume &conservedVariables,
                                 rvec3 &waveSpeed,
                                 bool computeWaveSpeed,
                                 volume::Volume &output)
{
    numericalFlux->computeFlux(conservedVariables, waveSpeed, computeWaveSpeed,
                               output);

    diffusionOperator->applyDiffusion(output, conservedVariables);
}


/// 
/// Returns the number of ghost cells needed.
/// This will take the maximum between the number of ghost cells the numerical
/// flux needs, and the number of ghost cells the diffusion operator needs
///
size_t ConservedSystem::getNumberOfGhostCells() const {
    return std::max(numericalFlux->getNumberOfGhostCells(),
        diffusionOperator->getNumberOfGhostCells());
}


}
}
