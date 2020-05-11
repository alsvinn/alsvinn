#include "alsfvm/python/PythonInterpreter.hpp"

namespace alsfvm {
namespace python {
PythonInterpreter& PythonInterpreter::getInstance() {
    static PythonInterpreter pythonInterpreter;
    return pythonInterpreter;
}

PythonInterpreter::~PythonInterpreter() {

    Py_Finalize();
}

PythonInterpreter::PythonInterpreter() {

    Py_Initialize();
    boost::python::numpy::initialize();
}
}
}

