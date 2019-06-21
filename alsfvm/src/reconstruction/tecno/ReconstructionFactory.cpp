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
#include "alsutils/config.hpp"
#include "alsfvm/reconstruction/tecno/ReconstructionFactory.hpp"
#include "alsfvm/reconstruction/tecno/ENOCPU.hpp"
#include "alsfvm/reconstruction/tecno/NoReconstruction.hpp"

#ifdef ALSVINN_HAVE_CUDA
    #include "alsfvm/reconstruction/tecno/ENOCUDA.hpp"
    #include "alsfvm/reconstruction/tecno/NoReconstructionCUDA.hpp"
#endif

namespace alsfvm {
namespace reconstruction {
namespace tecno {
alsfvm::shared_ptr<TecnoReconstruction> ReconstructionFactory::createReconstruction(
    const std::string& name,
    const std::string& equation,
    const simulator::SimulatorParameters& simulatorParameters,
    alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
    const grid::Grid& grid,
    alsfvm::shared_ptr<DeviceConfiguration>& deviceConfiguration
) {

    auto platform = deviceConfiguration->getPlatform();
    alsfvm::shared_ptr<TecnoReconstruction> reconstructor;

    if (platform == "cpu") {
        if (name == "none") {
            reconstructor.reset(new NoReconstruction);
        } else if (name == "eno2") {
            reconstructor.reset(new ENOCPU<2>(memoryFactory, grid.getDimensions().x,
                    grid.getDimensions().y,
                    grid.getDimensions().z));

        } else if (name == "eno3") {
            reconstructor.reset(new ENOCPU<3>(memoryFactory, grid.getDimensions().x,
                    grid.getDimensions().y,
                    grid.getDimensions().z));

        } else if (name == "eno4") {
            reconstructor.reset(new ENOCPU<4>(memoryFactory, grid.getDimensions().x,
                    grid.getDimensions().y,
                    grid.getDimensions().z));

        }

#ifdef ALSVINN_HAVE_CUDA
    } else if (platform == "cuda") {
        if (name == "none") {
            reconstructor.reset(new NoReconstructionCUDA);
        } else if (name == "eno2") {
            reconstructor.reset(new ENOCUDA<2>(memoryFactory, grid.getDimensions().x,
                    grid.getDimensions().y,
                    grid.getDimensions().z));

        } else if (name == "eno3") {
            reconstructor.reset(new ENOCUDA<3>(memoryFactory, grid.getDimensions().x,
                    grid.getDimensions().y,
                    grid.getDimensions().z));

        } else if (name == "eno4") {
            reconstructor.reset(new ENOCUDA<4>(memoryFactory, grid.getDimensions().x,
                    grid.getDimensions().y,
                    grid.getDimensions().z));

        }

#endif
    } else {
        THROW("Unknown platform " << platform);
    }


    if (!reconstructor.get()) {
        THROW("Something went wrong. Parameters were:\n"
            << "platfrom = " << platform << "\n"
            << "name = " << name << "\n");
    }

    return reconstructor;

}

}
}
}
