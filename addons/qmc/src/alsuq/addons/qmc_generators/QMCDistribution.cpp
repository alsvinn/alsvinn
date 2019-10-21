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

#include <alsutils/config.hpp>
#include "alsuq/addons/qmc_generators/QMCDistribution.hpp"

#ifdef ALSVINN_USE_MPI
    #include <mpi.h>
#endif
namespace alsuq {
namespace addons {
namespace qmc_generators {



QMCDistribution::QMCDistribution(size_t numberOfSamples, size_t dimension,
    std::function<void* ()> makeParametersFunction,
    std::function<void(void*, const char*, const char*)> setParameterFunction,
    std::function<void(void*)> deleteParametersFunction,
    std::function<void* (int, int, void*)> createFunction,
    std::function<void(void*)> deleteFunction,
    std::function<real(void*, int, int, int, int, void*)> generatorFunction,
    const alsutils::parameters::Parameters& parameters)
    : size(numberOfSamples),
      dimension(int(dimension)),
      samples(dimension, 0),
      generatorFunction(generatorFunction),
      deleteFunction(deleteFunction),
      qmcData(nullptr, deleteFunction),
      deleteParametersFunction(deleteParametersFunction),
      parametersStruct(makeParametersFunction(), deleteParametersFunction) {



#ifdef ALSVINN_USE_MPI
    int mpiNode;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiNode);
    setParameterFunction(parametersStruct.get(), "mpi_node",
        std::to_string(mpiNode).c_str());

    int mpiSize;
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    setParameterFunction(parametersStruct.get(), "mpi_size",
        std::to_string(mpiSize).c_str());
#else
    setParameterFunction(parametersStruct.get(), "mpi_node", "0");
    setParameterFunction(parametersStruct.get(), "mpi_size", "1");
#endif

    for (auto key : parameters.getKeys()) {

        setParameterFunction(parametersStruct.get(), key.c_str(),
            parameters.getString(key).c_str());
    }



    qmcData.reset(createFunction(size, dimension, parametersStruct.get()));



}

QMCDistribution::~QMCDistribution() {
    //empty
}

real QMCDistribution::operator()(
    size_t component, size_t sample) {
    return generatorFunction(qmcData.get(), size, dimension, component,
            sample, parametersStruct.get());
}

}
}
}
