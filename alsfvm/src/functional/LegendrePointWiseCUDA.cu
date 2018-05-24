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

#include "alsfvm/functional/LegendrePointWiseCUDA.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/functional/register_functional.hpp"
#include "alsutils/math/legendre.hpp"


namespace alsfvm {
namespace functional {

namespace {

template<int degree>
__global__ void legendreInnerKernel(memory::View<real> output,
                               memory::View<const real> input,
                               int ngx, int ngy, int ngz,
                               real weight, real minValue, real maxValue)
{
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if ( index >= output.size()) {
        return;
    }

    const size_t x = index % output.getNumberOfXCells();
    const size_t y = (index / output.getNumberOfXCells()) % output.getNumberOfYCells();
    const size_t z = index / (output.getNumberOfXCells()*output.getNumberOfYCells());
    const real value = input.at(x + ngx, y + ngy, z + ngz);
    const real scaledValue = (value-minValue) / (maxValue - minValue);
    output.at(index) += weight*alsutils::math::legendre<degree>(scaledValue);
}

template<int degree>
void runLegendre(memory::View<real> viewOut, memory::View<const real> viewIn,
                 ivec3 ghostCells, real weight,
                 real minValue, real maxValue) {

    const size_t threads = 1024;
    const size_t size = viewOut.size();
    legendreInnerKernel<degree><<<(size + threads -1)/threads, threads>>>(viewOut, viewIn,
                                                            ghostCells.x,
                                                            ghostCells.y,
                                                            ghostCells.z,
                                                            weight,
                                                             minValue,
                                                             maxValue);

}

void runLegendre( memory::View<real> viewOut, memory::View<const real> viewIn,
                 ivec3 ghostCells, real weight,
                 int degree,
                 real minValue, real maxValue) {
    switch (degree) {
    case 0:
        runLegendre<0>(viewOut, viewIn, ghostCells, weight, minValue, maxValue);
        break;
    case 1:
        runLegendre<1>(viewOut, viewIn, ghostCells, weight, minValue, maxValue);
        break;
    case 2:
        runLegendre<2>(viewOut, viewIn, ghostCells, weight, minValue, maxValue);
        break;
    case 3:
        runLegendre<3>(viewOut, viewIn, ghostCells, weight, minValue, maxValue);
        break;
    case 4:
        runLegendre<4>(viewOut, viewIn, ghostCells, weight, minValue, maxValue);
        break;
    case 5:
        runLegendre<5>(viewOut, viewIn, ghostCells, weight, minValue, maxValue);
        break;
    case 6:
        runLegendre<6>(viewOut, viewIn, ghostCells, weight, minValue, maxValue);
        break;
    case 7:
        runLegendre<7>(viewOut, viewIn, ghostCells, weight, minValue, maxValue);
        break;
    case 8:
        runLegendre<8>(viewOut, viewIn, ghostCells, weight, minValue, maxValue);
        break;
    case 9:
        runLegendre<9>(viewOut, viewIn, ghostCells, weight, minValue, maxValue);
        break;
    case 10:
        runLegendre<10>(viewOut, viewIn, ghostCells, weight, minValue, maxValue);
        break;
    case 11:
        runLegendre<11>(viewOut, viewIn, ghostCells, weight, minValue, maxValue);
        break;
    default:
        THROW("We have not specialized for Legendre polynomials of degree " << degree
              << "\nMaximum degree: 11.");
    }
}

}

LegendrePointWiseCUDA::LegendrePointWiseCUDA(const Functional::Parameters& parameters)
    : minValue(parameters.getDouble("minValue")),
      maxValue(parameters.getDouble("maxValue")),
      degree(parameters.getInteger("degree")) {
    if (parameters.contains("variables")) {
        for (auto variable : parameters.getStringVectorFromString("variables")) {
            variables.push_back(variable);
        }
    }
}

void LegendrePointWiseCUDA::operator()(volume::Volume& conservedVolumeOut,
    volume::Volume& extraVolumeOut,
    const volume::Volume& conservedVolumeIn,
    const volume::Volume& extraVolumeIn,
    const real weight,
    const grid::Grid& grid) {


    if (variables.size() == 0) {
        for (size_t var = 0; var < conservedVolumeIn.getNumberOfVariables(); ++var) {
            variables.push_back(conservedVolumeIn.getName(var));
        }

        for (size_t var = 0; var < extraVolumeIn.getNumberOfVariables(); ++var) {
            variables.push_back(extraVolumeIn.getName(var));
        }

    }

    const auto lengths = grid.getCellLengths();


    const auto ghostCells = conservedVolumeIn.getNumberOfGhostCells();

    const auto innerSize = conservedVolumeIn.getInnerSize();

    for (const std::string& variableName : variables) {
        if (conservedVolumeIn.hasVariable(variableName)) {


            auto viewIn = conservedVolumeIn.getScalarMemoryArea(variableName)->getView();
            auto viewOut = conservedVolumeOut.getScalarMemoryArea(variableName)->getView();


            runLegendre(viewOut, viewIn, ghostCells, weight, degree, minValue, maxValue);


        } else if (extraVolumeIn.hasVariable(variableName)) {

            auto viewIn = extraVolumeIn.getScalarMemoryArea(variableName)->getView();
            auto viewOut = extraVolumeOut.getScalarMemoryArea(variableName)->getView();

            runLegendre(viewOut, viewIn, ghostCells, weight, degree, minValue, maxValue);



        } else {
            THROW("Unknown variable name given to LegendrePointWiseCUDA functional: " <<
                variableName);
        }
    }

}

ivec3 LegendrePointWiseCUDA::getFunctionalSize(const grid::Grid& grid) const {
    return grid.getDimensions();
}

REGISTER_FUNCTIONAL(cuda, legendre_pointwise, LegendrePointWiseCUDA)
}
}
