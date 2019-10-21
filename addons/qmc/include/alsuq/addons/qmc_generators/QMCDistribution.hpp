/* Copyright (c) 2019 ETH Zurich, Kjetil Olsen Lye
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

#pragma once

#include <functional>
#include <alsutils/parameters/Parameters.hpp>
#include <alsutils/types.hpp>
#include <functional>

namespace alsuq {
namespace addons {
namespace qmc_generators {
using alsutils::real;

class QMCDistribution {
public:

    QMCDistribution(size_t numberOfSamples, size_t dimension,
        std::function<void* ()> makeParametersFunction,
        std::function<void(void*, const char*, const char*)> setParameterFunction,
        std::function<void(void*)> deleteParametersFunction,
        std::function<void* (int, int, void*)> createFunction,
        std::function<void(void*)> deleteFunction,
        std::function<real(void*, int, int, int, int, void*)> generatorFunction,
        const alsutils::parameters::Parameters& parameters);
    virtual ~QMCDistribution();

    //! Generates the next number from the QMC generator
    virtual real operator()(size_t component,
        size_t sample);

    static std::string getClassName() {
        return "QMCDistribution";
    }


private:

    const int size = 0;
    const int dimension = 0;
    std::vector<int> samples;

    using QMCData = void*;
    using QMCDataDeleter = std::function < void(QMCData)>;

    std::function<real(QMCData, int, int, int, int, void*)> generatorFunction;
    std::function<void(QMCData)> deleteFunction;
    std::unique_ptr<void, QMCDataDeleter> qmcData;


    std::function < void(QMCData)> deleteParametersFunction;
    std::unique_ptr<void, QMCDataDeleter> parametersStruct;
};
}
}
}
