/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "alsfvm/simulator/ConservedSystem.hpp"
#include <algorithm>
#include <thread>
#include "alsutils/timer/Timer.hpp"

namespace alsfvm {
namespace simulator {

ConservedSystem::ConservedSystem(alsfvm::shared_ptr<numflux::NumericalFlux>
    numericalFlux,
    alsfvm::shared_ptr<diffusion::DiffusionOperator> diffusionOperator)
    : numericalFlux(numericalFlux), diffusionOperator(diffusionOperator) {

}

void ConservedSystem::operator()( volume::Volume& conservedVariables,
    rvec3& waveSpeed,
    bool computeWaveSpeed,
    volume::Volume& output) {
    // We do this a bit "weird". In one thread we do the halo exchange
    // AND compute the sides, in the other thread, we compute the inner region.


    const auto ghostCells = output.getNumberOfGhostCells();
    const auto size = output.getTotalDimensions();
    const int dimensions = output.getDimensions();
    rvec3 waveSpeedSides({0, 0, 0});


    std::thread cellExchangeThread([&]() {
        if (cellExchanger) {

            ALSVINN_TIME_BLOCK(alsvinn, mpi, exchange);
            auto request = cellExchanger->exchangeCells(conservedVariables,
                    conservedVariables);
            {
                ALSVINN_TIME_BLOCK(alsvinn, mpi, exchange, wait);
                request.waitForAll();
            }

        }
    });

    numericalFlux->computeFlux(conservedVariables, waveSpeed, computeWaveSpeed,
        output, ghostCells, -1 * ghostCells);


    cellExchangeThread.join();

    // Now compute the sides
    for (int d = 0; d < dimensions; ++d) {

        ivec3 startBottom = {0, 0, 0};
        ivec3 endBottom = {0, 0, 0};


        endBottom [d] = -size[d] + 3 * ghostCells[d];
        rvec3 waveSpeedBottom = 0;
        numericalFlux->computeFlux(conservedVariables, waveSpeedBottom,
            computeWaveSpeed,
            output,
            startBottom, endBottom);

        for (int k = 0; k < 3; ++k) {
            waveSpeedSides[k] = std::max(waveSpeedSides[k], waveSpeedBottom[k]);
        }

        ivec3 startTop = {0, 0, 0};
        startTop[d] = size[d] - 3 * ghostCells[d];

        ivec3 endTop = {0, 0, 0};
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

void ConservedSystem::setCellExchanger(mpi::CellExchangerPtr cellExchanger) {
    this->cellExchanger = cellExchanger;
}


}
}
