
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

#include "alsfvm/io/PythonScript.hpp"
#ifdef _DEBUG
    #undef _DEBUG
    #include <Python.h>

    #define _DEBUG
#else
    #include <Python.h>
#endif

#include "alsfvm/types.hpp"
#include <iostream>
#include <sstream>
#include "alsfvm/volume/volume_foreach.hpp"
#include "alsfvm/python/PythonInterpreter.hpp"
#include "alsutils/log.hpp"
#include <fstream>
#include <iostream>
#include "alsfvm/python/handle_pyerror.hpp"
#ifdef ALSVINN_USE_MPI
    #include "alsutils/mpi/safe_call.hpp"
#endif
namespace alsfvm {
namespace io {

PythonScript::PythonScript(const std::string& basename,
    const Parameters& parameters, alsutils::mpi::ConfigurationPtr mpiConfiguration)

    :
    pythonCode(parameters.getString("pythonCode")),
    pythonClass(parameters.getString("pythonClass")),
    pythonInterpreterInstance(python::PythonInterpreter::getInstance()),
    mpiConfiguration(mpiConfiguration)

{

    try {
        boost::python::object mainModule = boost::python::import("__main__");
        boost::python::object mainNamespace = mainModule.attr("__dict__");

        boost::python::exec(pythonCode.c_str(), mainNamespace);

        boost::python::dict parametersAsDictionary;

        for (const auto& key : parameters.getKeys()) {
            parametersAsDictionary[key] = parameters.getString(key);
        }


        classInstance = mainNamespace[pythonClass](parametersAsDictionary);

    } catch (boost::python::error_already_set&) {
        HANDLE_PYTHON_EXCEPTION
    }
}

void PythonScript::write(const volume::Volume& conservedVariables,
    const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {

    try {
        copyToDatasets(conservedVariables);
        classInstance.attr("write")(datasetsConserved,
            makeGrid(grid));


#ifdef ALSVINN_USE_MPI

        if (mpiConfiguration) {
            MPI_SAFE_CALL(MPI_Barrier(mpiConfiguration->getCommunicator()));
        }

#endif

        classInstance.attr("afterMPISync")();

    } catch (boost::python::error_already_set&) {
        HANDLE_PYTHON_EXCEPTION
    }
}

void PythonScript::finalize(const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {

    classInstance.attr("finalize")();
}

void PythonScript::makeDatasets(const volume::Volume& conservedVariables) {
    rawPointersConserved.resize(0);
    auto size = conservedVariables.getInnerSize();

    boost::python::tuple shape = boost::python::make_tuple(size.x * size.y *
            size.z);
    boost::python::tuple stride = boost::python::make_tuple(sizeof(real));
    auto type = boost::python::numpy::dtype::get_builtin<real>();


    for (size_t var = 0; var < conservedVariables.getNumberOfVariables(); ++var) {

        auto array = boost::python::numpy::zeros(shape, type);
        rawPointersConserved.push_back((real*)array.get_data());
        datasetsConserved[conservedVariables.getName(var)] = array;


    }

    datasetsInitialized = true;
}

void PythonScript::copyToDatasets(const volume::Volume& conservedVariables) {

    if (!datasetsInitialized) {
        makeDatasets(conservedVariables);
    }

    const auto size = conservedVariables.getInnerSize();

    const auto totalSize = size.x * size.y * size.z;

    for (size_t var = 0; var < conservedVariables.getNumberOfVariables(); ++var) {

        conservedVariables.copyInternalCells(var,
            rawPointersConserved[var], totalSize);

    }

}

boost::python::api::object PythonScript::makeGrid(const grid::Grid& grid) {
    boost::python::dict gridAsDict;
    gridAsDict["origin"] = boost::python::make_tuple(grid.getOrigin().x,
            grid.getOrigin().y, grid.getOrigin().z);

    gridAsDict["top"] = boost::python::make_tuple(grid.getTop().x,
            grid.getTop().y, grid.getTop().z);

    gridAsDict["global_size"] = boost::python::make_tuple(grid.getGlobalSize().x,
            grid.getGlobalSize().y, grid.getGlobalSize().z);

    gridAsDict["local_size"] = boost::python::make_tuple(grid.getDimensions().x,
            grid.getDimensions().y, grid.getDimensions().z);
    gridAsDict["global_position"] = boost::python::make_tuple(
            grid.getGlobalPosition().x,
            grid.getGlobalPosition().y, grid.getGlobalPosition().z);

    return gridAsDict;

}

}
}
