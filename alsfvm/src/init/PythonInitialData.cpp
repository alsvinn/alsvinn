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
namespace alsfvm { namespace init {

namespace {


    // Adds indent to each line, eg transfer
    // "a\nb\n" to
    // "    a\n    b\n"
    void addIndent(const std::string& inputString, std::stringstream& output) {
        std::istringstream userSpecifiedFunction(inputString);

        std::string line;
        while (std::getline(userSpecifiedFunction, line)) {
            output <<"    "<< line << std::endl;
        }
    }
}
PythonInitialData::PythonInitialData(const std::string &programString, const Parameters& parameters)
    : parameters(parameters), programString(programString)
{

}

void PythonInitialData::setInitialData(volume::Volume& conservedVolume,
                                       volume::Volume& extraVolume,
                                       volume::Volume& primitiveVolume,
                                       equation::CellComputer& cellComputer,
                                       grid::Grid& grid)
{

    auto& interpreter = PythonInterpreter::getInstance();

    // We need to add namespaces for the function to live in, this way
    // we can actually reference later.


    auto globalNamespace = interpreter.getGlobalNamespace();
    PyObject* moduleLocal(PyImport_AddModule("__main__"));
    PyObject* localNamespace(PyModule_GetDict(moduleLocal));
    CHECK_PYTHON
    // This will hold the inputs for our function. We allocate this once,
    // and use it several times.
    PythonObjectHolder argumentTuple(PyTuple_New(7));


    // Now we declare the wrappers around the function.
    std::stringstream functionStringStream;

    // We need to figure out if we are dealing with a function or just a
    // snippet
    bool snippet = true;
    if (programString.find("init_global") != std::string::npos) {

        snippet = false;
    }

    functionStringStream << "from math import *" << std::endl;
    functionStringStream << "try:\n    from numpy import *\nexcept:\n    pass" << std::endl;
    functionStringStream << "def initial_data(x, y, z, i, j, k, output):\n";

    // Now we need to add the variables we need to write:
    for(size_t i = 0; i < primitiveVolume.getNumberOfVariables(); ++i) {
        // We set them to None, that way we can check at the end if they are checked.
        addIndent(primitiveVolume.getName(i) + " = 0.0", functionStringStream);
    }

    // Then we add the parameters
    for (auto parameterName : parameters.getParameterNames()) {
        auto parameter = parameters.getParameter(parameterName);
        // See http://stackoverflow.com/questions/3001239/define-a-global-in-a-python-module-from-a-c-api
        // and https://docs.python.org/3/c-api/object.html#PyObject_SetAttrString
        if (parameter.size() == 1) {
            PyObject* parameterValue = PyFloat_FromDouble(parameter[0]);
            PyObject_SetAttrString(moduleLocal, parameterName.c_str(), parameterValue);
            Py_DECREF(parameterValue);
            //addIndent(parameterName + " = " + std::to_string(parameter[0]), functionStringStream);
        }
        else {
            PyObject* parameterValueList = PyList_New(parameter.size());
           
            for (size_t i = 0; i < parameter.size(); ++i) {
                

                PyObject* parameterValue = PyFloat_FromDouble(parameter[i]);
                PyList_SetItem(parameterValueList, i, parameterValue);
                //Py_DECREF(parameterValue);
            }

            PyObject_SetAttrString(moduleLocal, parameterName.c_str(), parameterValueList);
            Py_DECREF(parameterValueList);
        }
    }
    if (snippet) {
        addIndent(programString, functionStringStream);


        // Add code to store the variables:
        for(size_t i = 0; i < primitiveVolume.getNumberOfVariables(); ++i) {
            // We set them to None, that way we can check at the end if they are checked.
            const auto& name = primitiveVolume.getName(i);
            addIndent(std::string("output['") + name + "'] = " + name, functionStringStream);
        }
    } else {
        // Add code to store the variables:
        for(size_t i = 0; i < primitiveVolume.getNumberOfVariables(); ++i) {
            // We set them to None, that way we can check at the end if they are checked.
            const auto& name = primitiveVolume.getName(i);
            addIndent(std::string("output['") + name + "'] = " + name+"_global[i,j,k]", functionStringStream);
        }
    }

    if (!snippet) {
        for(size_t i = 0; i < primitiveVolume.getNumberOfVariables(); ++i) {
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
        for(size_t i = 0; i < primitiveVolume.getNumberOfVariables(); ++i) {
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

    PyRun_String(functionStringStream.str().c_str(),
                 Py_file_input, globalNamespace, localNamespace);
    CHECK_PYTHON


    PythonObjectHolder initialValueFunction(PyObject_GetAttrString(moduleLocal, "initial_data"));


    // loop through the map and set the initial values
    volume::for_each_midpoint(primitiveVolume, grid,
                              [&](real x, real y, real z, size_t index,
                              size_t i, size_t j, size_t k) {

		if (PyErr_Occurred()) {
			PyErr_Print();
			THROW("Python error occured");
		}
        PyObject* outputMap(PyDict_New());

        PyObject* xObject(PyFloat_FromDouble(x));
        PyObject* yObject(PyFloat_FromDouble(y));
        PyObject* zObject(PyFloat_FromDouble(z));

        PyObject* iObject(PyLong_FromSize_t(i));
        PyObject* jObject(PyLong_FromSize_t(j));
        PyObject* kObject(PyLong_FromSize_t(k));

        CHECK_PYTHON

        PyTuple_SetItem(argumentTuple.object, 0, xObject);
        PyTuple_SetItem(argumentTuple.object, 1, yObject);
        PyTuple_SetItem(argumentTuple.object, 2, zObject);

        PyTuple_SetItem(argumentTuple.object, 3, iObject);
        PyTuple_SetItem(argumentTuple.object, 4, jObject);
        PyTuple_SetItem(argumentTuple.object, 5, kObject);
        PyTuple_SetItem(argumentTuple.object, 6, outputMap);

        CHECK_PYTHON
        PyObject_CallObject(initialValueFunction.object,
                                               argumentTuple.object);
        CHECK_PYTHON
        // Loop through each variable and set it in the primitive variables:
        for(size_t var = 0; var <  primitiveVolume.getNumberOfVariables(); ++var) {
            const auto& name = primitiveVolume.getName(var);
            PyObject* floatObject(PyDict_GetItemString(outputMap, name.c_str()));
            const double value = PyFloat_AsDouble(floatObject);

            primitiveVolume.getScalarMemoryArea(var)->getPointer()[index] = value;
        }
        CHECK_PYTHON
    });


    CHECK_PYTHON
            cellComputer.computeFromPrimitive(primitiveVolume, conservedVolume, extraVolume);
}

void PythonInitialData::setParameters(const Parameters &parameters)
{

    this->parameters = parameters;

}



}
}
