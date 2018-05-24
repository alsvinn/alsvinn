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

#include "alsfvm/functional/IdentityCUDA.hpp"
#include "alsfvm/functional/register_functional.hpp"

namespace alsfvm {
namespace functional {

namespace {

__global__ void addInnerKernel(memory::View<real> output,
                               memory::View<const real> input,
                               int ngx, int ngy, int ngz,
                               double weight)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if ( index >= output.size()) {
        return;
    }

    const size_t x = index % output.getNumberOfXCells();
    const size_t y = (index / output.getNumberOfXCells()) % output.getNumberOfYCells();
    const size_t z = index / (output.getNumberOfXCells()*output.getNumberOfYCells());
    output.at(index) += weight*input.at(x + ngx, y + ngy, z + ngz);
}

}
IdentityCUDA::IdentityCUDA(const Functional::Parameters& parameters) {
    if (parameters.contains("variables")) {
        for (auto variable : parameters.getStringVectorFromString("variables")) {
            variables.push_back(variable);
        }
    }
}

void IdentityCUDA::operator()(volume::Volume& conservedVolumeOut,
    volume::Volume& extraVolumeOut,
    const volume::Volume& conservedVolumeIn,
    const volume::Volume& extraVolumeIn,
    const real weight,
    const grid::Grid& ) {

    if (variables.size() == 0) {
        for (size_t var = 0; var < conservedVolumeIn.getNumberOfVariables(); ++var) {
            variables.push_back(conservedVolumeIn.getName(var));
        }

        for (size_t var = 0; var < extraVolumeIn.getNumberOfVariables(); ++var) {
            variables.push_back(extraVolumeIn.getName(var));
        }

    }


    const auto ghostCells = conservedVolumeIn.getNumberOfGhostCells();
    for (const std::string& variableName : variables) {
        if (conservedVolumeIn.hasVariable(variableName)) {



            auto viewIn = conservedVolumeIn.getScalarMemoryArea(variableName)->getView();
            auto viewOut = conservedVolumeOut.getScalarMemoryArea(variableName)->getView();

            const size_t threads = 1024;
            const size_t size = viewOut.size();
            addInnerKernel<<<(size + threads -1)/threads, threads>>>(viewOut, viewIn,
                                                                    ghostCells.x,
                                                                    ghostCells.y,
                                                                    ghostCells.z,
                                                                    weight);

        } else if (extraVolumeIn.hasVariable(variableName)) {

            auto viewIn = extraVolumeIn.getScalarMemoryArea(variableName)->getView();
            auto viewOut = extraVolumeOut.getScalarMemoryArea(variableName)->getView();

            const size_t threads = 1024;
            const size_t size = viewOut.size();
            addInnerKernel<<<(size + threads -1)/threads, threads>>>(viewOut, viewIn,
                                                                    ghostCells.x,
                                                                    ghostCells.y,
                                                                    ghostCells.z,
                                                                    weight);

        } else {
            THROW("Unknown variable name given to IdentityCUDA functional: " <<
                            variableName);
        }

    }
}

ivec3 IdentityCUDA::getFunctionalSize(const grid::Grid& grid) const {
    return grid.getDimensions();
}
REGISTER_FUNCTIONAL(cuda, identity, IdentityCUDA)
}
}
