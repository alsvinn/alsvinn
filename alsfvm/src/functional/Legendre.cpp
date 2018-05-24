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

#include "alsfvm/functional/Legendre.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/functional/register_functional.hpp"
#include <boost/math/special_functions/legendre.hpp>


namespace alsfvm {
namespace functional {

Legendre::Legendre(const Functional::Parameters& parameters)
    : minValue(parameters.getDouble("minValue")),
      maxValue(parameters.getDouble("maxValue")),
      degree_k(parameters.getInteger("degree_k")),
      degree_n(parameters.getInteger("degree_n")),
      degree_m(parameters.getInteger("degree_m")) {
    if (parameters.contains("variables")) {
        for (auto variable : parameters.getStringVectorFromString("variables")) {
            variables.push_back(variable);
        }
    }
}

void Legendre::operator()(volume::Volume& conservedVolumeOut,
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

    if (lengths.z > 1 || lengths.y == 1) {
        THROW("For now, Legendre polynomials only support 2d, givne dimensions " <<
            lengths);
    }

    const real dxdydz = lengths.x * lengths.y * lengths.z;


    const auto origin = grid.getOrigin();
    const auto top = grid.getTop();
    const auto sides = top - origin;

    for (const std::string& variableName : variables) {
        if (conservedVolumeIn.hasVariable(variableName)) {

            real integral = 0.0;

            volume::for_each_midpoint(conservedVolumeIn, grid, [&](real x, real y, real,
            size_t i) {

                // Scale from -1 to 1
                const auto xScaled =  2 * (x - origin.x) / sides.x - 1;
                const auto yScaled =  2 * (y - origin.y) / sides.y - 1;

                const real value = (conservedVolumeIn.getScalarMemoryArea(
                            variableName)->getPointer()[i] - minValue) / (maxValue - minValue);
                integral += boost::math::legendre_p(degree_k,
                        xScaled) * boost::math::legendre_p(degree_n, yScaled)
                    * boost::math::legendre_p(degree_m, value) * dxdydz;
            });

            conservedVolumeOut.getScalarMemoryArea(variableName)->getPointer()[0] += weight
                * integral;
        } else if (extraVolumeIn.hasVariable(variableName)) {

            real integral = 0.0;

            volume::for_each_midpoint(conservedVolumeIn, grid, [&](real x, real y, real,
            size_t i) {

                // Scale from -1 to 1
                const auto xScaled =  2 * (x - origin.x) / sides.x - 1;
                const auto yScaled =  2 * (y - origin.y) / sides.y - 1;
                const real value = (extraVolumeIn.getScalarMemoryArea(
                            variableName)->getPointer()[i] - minValue) / (maxValue - minValue);
                integral += boost::math::legendre_p(degree_k,
                        xScaled) * boost::math::legendre_p(degree_n, yScaled)
                    * boost::math::legendre_p(degree_m, value) * dxdydz;
            });

            extraVolumeOut.getScalarMemoryArea(variableName)->getPointer()[0] += weight *
                integral;

        } else {
            THROW("Unknown variable name given to Legendre functional: " << variableName);
        }
    }

}

ivec3 Legendre::getFunctionalSize(const grid::Grid& grid) const {
    return {1, 1, 1};
}

REGISTER_FUNCTIONAL(cpu, legendre, Legendre)
}
}
