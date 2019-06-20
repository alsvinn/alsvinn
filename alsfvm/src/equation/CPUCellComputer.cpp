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

#include "alsfvm/equation/CPUCellComputer.hpp"
#include "alsfvm/equation/euler/Euler.hpp"
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/equation/equation_list.hpp"

namespace alsfvm {
namespace equation {

template<class Equation> CPUCellComputer<Equation>::CPUCellComputer(
    simulator::SimulatorParameters& parameters)
    : parameters(static_cast<typename Equation::Parameters&>
          (parameters.getEquationParameters())) {

}

template<class Equation>
void CPUCellComputer<Equation>::computeExtraVariables(const volume::Volume&
    conservedVariables,
    volume::Volume& extraVariables) {
    Equation eq(parameters);
    volume::transform_volume<typename Equation::ConservedVariables,
           typename Equation::ExtraVariables>(
               conservedVariables, extraVariables,
               [&](const typename Equation::ConservedVariables & in)
    -> typename Equation::ExtraVariables {
        return eq.computeExtra(in);
    });
}


template<class Equation>
real CPUCellComputer<Equation>::computeMaxWaveSpeed(const volume::Volume&
    conservedVariables, size_t direction) {
    Equation eq(parameters);
    real maxWaveSpeed = 0;
    assert(direction < 3);
    volume::for_each_cell<typename Equation::ConservedVariables>(conservedVariables,
        [&maxWaveSpeed, direction,
                   &eq](const typename Equation::ConservedVariables & conserved, size_t index) {
        auto extra = eq.computeExtra(conserved);

        if (direction == 0) {
            const real waveSpeedX = eq.template computeWaveSpeed<0>(conserved, extra);
            maxWaveSpeed = std::max(maxWaveSpeed, waveSpeedX);
        } else if (direction == 1) {
            const real waveSpeedY = eq.template computeWaveSpeed<1>(conserved, extra);
            maxWaveSpeed = std::max(maxWaveSpeed, waveSpeedY);
        }    else if (direction == 2) {
            const real waveSpeedZ = eq.template computeWaveSpeed<2>(conserved, extra);
            maxWaveSpeed = std::max(maxWaveSpeed, waveSpeedZ);
        }
    });

    return maxWaveSpeed;
}

///
/// Checks if all the constraints for the equation are met
/// \param conservedVariables the conserved variables (density, momentum, Energy for Euler)
/// \param extraVariables the extra variables (pressure and velocity for Euler)
/// \return true if it obeys the constraints, false otherwise
/// \todo Tidy up the way we check for nan and inf
///
template<class Equation>
bool CPUCellComputer<Equation>::obeysConstraints(const volume::Volume&
    conservedVariables) {

    bool obeys = true;
    Equation eq(parameters);

    volume::for_each_cell<typename Equation::ConservedVariables>(conservedVariables,
        [&obeys,
            &eq](const typename Equation::ConservedVariables & conserved,
    size_t index) {
        const auto extra = eq.computeExtra(conserved);
        // Check for nan and inf:
        const real* conservedAsRealPtr = (const real*)&conserved;

        for (size_t i = 0; i < sizeof(conserved) / sizeof(real); i++) {
            if (std::isnan(conservedAsRealPtr[i]) || std::isinf(conservedAsRealPtr[i])) {
                obeys = false;
                std::cerr << "Component (conserved)" << i << " isnan or inf" << " at index " <<
                    index << std::endl;
                std::cerr << "values are: " << std::endl;

                for (size_t j = 0; j < sizeof(conserved) / sizeof(real); j++) {
                    std::cout << conservedAsRealPtr[j] << std::endl;
                }

                std::cout << std::endl << std::endl;
            }
        }


        const real* extraAsRealPtr = (const real*)&extra;

        for (size_t i = 0; i < sizeof(extra) / sizeof(real); i++) {
            if (std::isnan(extraAsRealPtr[i]) || std::isinf(extraAsRealPtr[i])) {
                obeys = false;
            }
        }

        if (!eq.obeysConstraints(conserved, extra)) {
            std::cerr << "Does not obey constraint at" << index << std::endl;
            obeys = false;
        }
    });

    return obeys;
}

template<class Equation>
void CPUCellComputer<Equation>::computeFromPrimitive(const volume::Volume&
    primitiveVariables,
    volume::Volume& conservedVariables) {
    Equation eq(parameters);

    volume::transform_volume<typename Equation::PrimitiveVariables,
           typename Equation::ConservedVariables>(
               primitiveVariables, conservedVariables,
               [&](const typename Equation::PrimitiveVariables & in)
    -> typename Equation::ConservedVariables {
        return eq.computeConserved(in);
    });
}

ALSFVM_EQUATION_INSTANTIATE(CPUCellComputer)
}
}
