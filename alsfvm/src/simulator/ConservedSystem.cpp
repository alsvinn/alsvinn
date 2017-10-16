#include "alsfvm/simulator/ConservedSystem.hpp"
#include <algorithm>
#include <thread>
namespace alsfvm { namespace simulator {

ConservedSystem::ConservedSystem(alsfvm::shared_ptr<numflux::NumericalFlux> numericalFlux,
    alsfvm::shared_ptr<diffusion::DiffusionOperator> diffusionOperator)
    : numericalFlux(numericalFlux), diffusionOperator(diffusionOperator)
{

}

void ConservedSystem::operator()( volume::Volume &conservedVariables,
                                 rvec3 &waveSpeed,
                                 bool computeWaveSpeed,
                                 volume::Volume &output)
{
    // We do this a bit "weird". In one thread we do the halo exchange
    // AND compute the sides, in the other thread, we compute the inner region.


    const auto ghostCells = output.getNumberOfGhostCells();
    const auto size = output.getTotalDimensions();
    const int dimensions = output.getDimensions();
    rvec3 waveSpeedSides({0,0,0});


    std::thread cellExchangeThread([&]() {
        if (cellExchanger) {
            cellExchanger->exchangeCells(conservedVariables, conservedVariables).waitForAll();
        }
    });

    numericalFlux->computeFlux(conservedVariables, waveSpeed, computeWaveSpeed,
                               output, ghostCells, -1*ghostCells);


    cellExchangeThread.join();


     // Now compute the sides
     for (int d = 0; d < dimensions; ++d) {

         ivec3 startBottom = {0,0,0};
         ivec3 endBottom = {0,0,0};


         endBottom [d] = -size[d]+3*ghostCells[d];
         rvec3 waveSpeedBottom = 0;
         numericalFlux->computeFlux(conservedVariables, waveSpeedBottom, computeWaveSpeed,
                                    output,
                                    startBottom, endBottom);
         for (int k = 0; k < 3; ++k) {
             waveSpeedSides[k] = std::max(waveSpeedSides[k], waveSpeedBottom[k]);
         }

         ivec3 startTop= {0, 0, 0};
         startTop[d] = size[d]-3 *ghostCells[d];

         ivec3 endTop= {0,0,0};
         rvec3 waveSpeedTop = 0;
         numericalFlux->computeFlux(conservedVariables, waveSpeedTop, computeWaveSpeed,
                                    output,
                                    startTop,
                                    endTop);

         for (int k = 0; k < 3; ++k) {
             waveSpeedSides[k] = std::max(waveSpeedSides[k], waveSpeedTop[k]);
         }
     }

     for (int k = 0; k < 3; ++k) {
         waveSpeed[k] = std::max(waveSpeed[k], waveSpeedSides[k]);
     }

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

void ConservedSystem::setCellExchanger(mpi::CellExchangerPtr cellExchanger)
{
    this->cellExchanger = cellExchanger;
}


}
}
