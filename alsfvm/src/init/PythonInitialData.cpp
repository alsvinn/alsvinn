#include "alsfvm/init/PythonInitialData.hpp"

namespace alsfvm { namespace init {

namespace {
    // Utility struct to act as a smart pointer
    struct PythonObjectHolder {
        PythonObjectHolder(PyObject* object)
            : object(object)
        {}

        ~PythonObjectHolder() {
            Py_DECREF(object);
        }

        PyObject* object;
    };
}
void PythonInitialData::setInitialData(volume::Volume &conservedVolume, volume::Volume &extraVolume, grid::Grid &grid)
{

    Py_Initialize();

    // First parse the function
    PythonObjectHolder globalNamespace(PyDict_New());
    PythonObjectHolder module(PyModule_New("alsfvm_init"));
    PythonObjectHolder localNamespace(PyModule_GetDict(module));
    PythonObjectHolder argumentTuple(PyTuple_New(3));
    PyRun_String("def function(x,y,z):"
                 "    print (\"%s %s %s\" % (x,y,z))",
                 Py_file_input, globalNamespace, localNamespace);

    PyObject *x = PyFloat_FromDouble(3);
    PyObject *y = PyFloat_FromDouble(2);
    PyObject *z = PyFloat_FromDouble(1);

    PyTuple_SetItem(pArgs, 0, x);
    PyTuple_SetItem(pArgs, 1, y);
    PyTuple_SetItem(pArgs, 2, z);

    PyObject* function = PyObject_GetAttrString(module, "function");
    PyObject_CallObject(function, pArgs);


    Py_Finalize();
}
}
}
