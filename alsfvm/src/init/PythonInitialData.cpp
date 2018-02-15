#include "alsfvm/init/PythonInitialData.hpp"

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
#define L std::cout << __FILE__ << ":" << __LINE__ << std::endl;

#define CHECK_PYTHON \
        if ( PyErr_Occurred()) { \
            PyErr_Print(); \
            THROW("Python error occured"); \
        }

using namespace alsfvm::python;
namespace alsfvm {
namespace init {

namespace {


// Adds indent to each line, eg transfer
// "a\nb\n" to
// "    a\n    b\n"
void addIndent(const std::string& inputString, std::stringstream& output) {
    std::istringstream userSpecifiedFunction(inputString);

    std::string line;

    while (std::getline(userSpecifiedFunction, line)) {
        output << "    " << line << std::endl;
    }
}
}
PythonInitialData::PythonInitialData(const std::string& programString,
    const Parameters& parameters)
    : parameters(parameters), programString(programString) {

}

void PythonInitialData::setInitialData(volume::Volume& conservedVolume,
    volume::Volume& extraVolume,
    volume::Volume& primitiveVolume,
    equation::CellComputer& cellComputer,
    grid::Grid& grid) {

    PythonInterpreter::getInstance();
    boost::python::object mainModule = boost::python::import("__main__");
    boost::python::object mainNamespace = mainModule.attr("__dict__");

    // Now we declare the wrappers around the function.
    std::stringstream functionStringStream;

    // We need to figure out if we are dealing with a function or just a
    // snippet
    bool snippet = true;

    if (programString.find("init_global") != std::string::npos) {

        snippet = false;
    }

    functionStringStream << "from math import *" << std::endl;
    functionStringStream << "try:\n    from numpy import *\nexcept:\n    pass" <<
        std::endl;
    functionStringStream << "def initial_data(x, y, z, i, j, k, output):\n";

    // Now we need to add the variables we need to write:
    for (size_t i = 0; i < primitiveVolume.getNumberOfVariables(); ++i) {
        // We set them to None, that way we can check at the end if they are checked.
        addIndent(primitiveVolume.getName(i) + " = 0.0", functionStringStream);
    }

    // Then we add the parameters
    for (auto parameterName : parameters.getParameterNames()) {
        const auto& parameter = parameters.getParameter(parameterName);

        // See http://stackoverflow.com/questions/3001239/define-a-global-in-a-python-module-from-a-c-api
        // and https://docs.python.org/3/c-api/object.html#PyObject_SetAttrString
        if (parameter.size() == 1) {
            mainModule.attr(parameterName.c_str()) = boost::python::object(parameter[0]);

        } else {
            boost::python::tuple shape = boost::python::make_tuple(parameter.size());
            boost::python::tuple stride = boost::python::make_tuple(sizeof(real));
            auto type = boost::python::numpy::dtype::get_builtin<real>();
            auto array = boost::python::numpy::from_data(parameter.data(),
                    type, shape, stride,
                    mainModule);
            mainModule.attr(parameterName.c_str()) = boost::python::object(array);

        }
    }

    if (snippet) {
        addIndent(programString, functionStringStream);


        // Add code to store the variables:
        for (size_t i = 0; i < primitiveVolume.getNumberOfVariables(); ++i) {
            // We set them to None, that way we can check at the end if they are checked.
            const auto& name = primitiveVolume.getName(i);
            addIndent(std::string("output['") + name + "'] = " + name,
                functionStringStream);
        }
    } else {
        // Add code to store the variables:
        for (size_t i = 0; i < primitiveVolume.getNumberOfVariables(); ++i) {
            // We set them to None, that way we can check at the end if they are checked.
            const auto& name = primitiveVolume.getName(i);
            addIndent(std::string("output['") + name + "'] = " + name + "_global[i,j,k]",
                functionStringStream);
        }
    }

    if (!snippet) {
        for (size_t i = 0; i < primitiveVolume.getNumberOfVariables(); ++i) {
            // We set them to None, that way we can check at the end if they are checked.
            const auto& name = primitiveVolume.getName(i);
            functionStringStream << name << "_global = zeros(("
                << grid.getDimensions().x << ", "
                << grid.getDimensions().y << ", "
                << grid.getDimensions().z << "))"
                << std::endl;
        }

        functionStringStream << programString << std::endl;

        functionStringStream << "init_global(";

        for (size_t i = 0; i < primitiveVolume.getNumberOfVariables(); ++i) {
            // We set them to None, that way we can check at the end if they are checked.
            const auto& name = primitiveVolume.getName(i);
            functionStringStream << name << "_global, ";
        }

        functionStringStream << grid.getDimensions().x << ", "
            << grid.getDimensions().y << ", "
            << grid.getDimensions().z << ")"
            << std::endl;
    }

    ALSVINN_LOG(INFO, "Python program: \n" << functionStringStream.str());

    boost::python::exec(functionStringStream.str().c_str(), mainNamespace);
    ALSVINN_LOG(INFO, "Pythonprogram executed");

    auto initialValueFunction = mainNamespace["initial_data"];

    // loop through the map and set the initial values
    volume::for_each_midpoint(primitiveVolume, grid,
        [&](real x, real y, real z, size_t index,
    size_t i, size_t j, size_t k) {



        boost::python::dict outputMap;

        initialValueFunction(x, y, z, i, j, k, outputMap);

        // Loop through each variable and set it in the primitive variables:
        for (size_t var = 0; var <  primitiveVolume.getNumberOfVariables(); ++var) {
            const auto& name = primitiveVolume.getName(var);

            const double value = boost::python::extract<double>(outputMap[name]);

            primitiveVolume.getScalarMemoryArea(var)->getPointer()[index] = value;
        }

    });

    ALSVINN_LOG(INFO, "Done setting initial data");
    cellComputer.computeFromPrimitive(primitiveVolume, conservedVolume,
        extraVolume);
}

void PythonInitialData::setParameters(const Parameters& parameters) {

    this->parameters = parameters;

}



}
}
