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

#include "alsuq/distribution/DLLDistribution.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/dll.hpp>
#ifdef ALSVINN_USE_MPI
    #include <mpi.h>
#endif
namespace alsuq {
namespace distribution {


DLLDistribution::DLLDistribution(size_t numberOfSamples, size_t dimension,
    const Parameters& parameters)
    : size(numberOfSamples), dimension(dimension), samples(dimension, 0) {
    auto filename = parameters.getString("library");
    auto createFunctionName = parameters.getString("create_function");
    auto makeParametersName = parameters.getString("make_parameters_function");

    if (boost::algorithm::to_lower_copy(makeParametersName) != "none") {
        auto makeParametersFunction = boost::dll::import <void* ()>(filename,
                makeParametersName);
        parametersStruct = makeParametersFunction();

        auto setParameterFunctionName = parameters.getString("set_parameter_function");

        auto setParameterFunction = boost::dll::import
            <void(void*, const char*, const char*)>(filename,
                setParameterFunctionName);

        auto deleteParametersFunctionName =
            parameters.getString("delete_parameters_function");

        deleteParametersFunction = boost::dll::import
            <void(void*)>(filename,
                deleteParametersFunctionName);


#ifdef ALSVINN_USE_MPI
        int mpiNode;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiNode);
        setParameterFunction(parametersStruct, "mpi_node",
            std::to_string(mpiNode).c_str());

        int mpiSize;
        MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
        setParameterFunction(parametersStruct, "mpi_size",
            std::to_string(mpiSize).c_str());
#else
        setParameterFunction(parametersStruct, "mpi_node", "0");
        setParameterFunction(parametersStruct, "mpi_size", "1");
#endif

        for (auto key : parameters.getKeys()) {

            setParameterFunction(parametersStruct, key.c_str(),
                parameters.getString(key).c_str());
        }

    }

    if (boost::algorithm::to_lower_copy(createFunctionName) != "none") {
        auto createFunction = boost::dll::import<void* (int, int, void*)>(filename,
                createFunctionName);
        dllData = createFunction(size, dimension, parametersStruct);

        auto deleteFunctionName = parameters.getString("delete_function");

        if (boost::algorithm::to_lower_copy(deleteFunctionName) != "none") {
            deleteFunction = boost::dll::import<void(void*)>(filename, deleteFunctionName);
        }
    }

    auto generatorFunctionName = parameters.getString("generator_function");
    generatorFunction = boost::dll::import<real(void*, int, int, int, int, void*)>
        (filename, generatorFunctionName);
}

DLLDistribution::~DLLDistribution() {
    if (deleteFunction) {
        deleteFunction(dllData);
    }

    if (deleteParametersFunction && parametersStruct) {
        deleteParametersFunction(parametersStruct);
    }
}

real DLLDistribution::generate(generator::Generator& generator,
    size_t component, size_t sample) {
    return generatorFunction(dllData, size, dimension, component,
            sample, parametersStruct);
}

}



}
