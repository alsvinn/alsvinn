#pragma once
#ifdef _DEBUG
    #undef _DEBUG
    #include <Python.h>
    #define _DEBUG
#else
    #include <Python.h>

#endif
#include <boost/python/numpy.hpp>

namespace alsfvm {
namespace python {


class PythonInterpreter {
public:
    static PythonInterpreter& getInstance() {
        static PythonInterpreter pythonInterpreter;
        return pythonInterpreter;
    }

    ~PythonInterpreter() {

        Py_Finalize();
    }

private:

    PythonInterpreter() {

        Py_Initialize();
        boost::python::numpy::initialize();

    }


};
}
}
