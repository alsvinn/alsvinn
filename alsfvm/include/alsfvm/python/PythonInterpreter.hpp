#pragma once
#include <Python.h>

namespace alsfvm {
namespace python {

// Utility struct to act as a smart pointer
struct PythonObjectHolder {
    PythonObjectHolder(PyObject* object)
        : object(object)
    {}

    ~PythonObjectHolder() {
        //Py_DECREF(object);
    }

    PyObject* object;

private:
    PythonObjectHolder(const PythonObjectHolder&) {
        // we don't want to copy this object
    }
};

class PythonInterpreter {
public:
    static PythonInterpreter& getInstance() {
        static PythonInterpreter pythonInterpreter;
        return pythonInterpreter;
    }

    ~PythonInterpreter() {
        Py_DECREF(globalNamespace);
        Py_Finalize();
    }

    PyObject* getGlobalModule() {
        return moduleGlobal;
    }

    PyObject* getGlobalNamespace() {
        return globalNamespace;
    }

private:
    PyObject* moduleGlobal;
    PyObject* globalNamespace;
    PythonInterpreter() {
        Py_Initialize();
        moduleGlobal = PyImport_AddModule("__main__");
        globalNamespace = PyModule_GetDict(moduleGlobal);
    }


};
}
}
