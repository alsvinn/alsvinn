
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
namespace alsfvm {
namespace io {

PythonScript::PythonScript(const std::string& basename,
    const Parameters& parameters)
    :
    pythonCode(parameters.getString("pythonCode")),
    pythonClass(parameters.getString("pythonClass")),
    pythonInterpreterInstance(python::PythonInterpreter::getInstance()) {

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
    const volume::Volume& extraVariables, const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {

    try {
        copyToDatasets(conservedVariables, extraVariables);
        classInstance.attr("write")(datasetsConserved,
            datasetsExtra);
    } catch (boost::python::error_already_set&) {
        HANDLE_PYTHON_EXCEPTION
    }
}

void PythonScript::finalize(const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {

    classInstance.attr("finalize")();
}

void PythonScript::makeDatasets(const volume::Volume& conservedVariables,
    const volume::Volume& extraVariables) {
    rawPointersConserved.resize(0);
    rawPointersExtra.resize(0);
    auto size = conservedVariables.getInnerSize();
    boost::python::tuple shape = boost::python::make_tuple(size.x, size.y, size.z);
    boost::python::tuple stride = boost::python::make_tuple(sizeof(real));
    auto type = boost::python::numpy::dtype::get_builtin<real>();


    for (size_t var = 0; var < conservedVariables.getNumberOfVariables(); ++var) {

        auto array = boost::python::numpy::zeros(shape, type);
        rawPointersConserved.push_back((double*)array.get_data());
        datasetsConserved[conservedVariables.getName(var)] = array;
    }

    for (size_t var = 0; var < extraVariables.getNumberOfVariables(); ++var) {
        auto array = boost::python::numpy::zeros(shape, type);
        rawPointersExtra.push_back((double*)array.get_data());
        datasetsExtra[extraVariables.getName(var)] = array;
    }

    datasetsInitialized = true;
}

void PythonScript::copyToDatasets(const volume::Volume& conservedVariables,
    const volume::Volume& extraVariables) {

    if (!datasetsInitialized) {
        makeDatasets(conservedVariables, extraVariables);
    }

    const auto size = conservedVariables.getInnerSize();
    const auto totalSize = size.x * size.y * size.z;

    for (size_t var = 0; var < conservedVariables.getNumberOfVariables(); ++var) {

        conservedVariables.copyInternalCells(var,
            rawPointersConserved[var], totalSize);

    }

    for (size_t var = 0; var < extraVariables.getNumberOfVariables(); ++var) {
        extraVariables.copyInternalCells(var,
            rawPointersExtra[var], totalSize);

    }
}

}
}
