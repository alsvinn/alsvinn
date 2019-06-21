#include "alsutils/get_python_version.hpp"
#include <sstream>
#include <Python.h>
namespace alsutils {
std::string getPythonVersion() {
    std::stringstream ss;
    ss << "Runtime: " << Py_GetVersion() << ", compile time: " << PY_VERSION;

    return ss.str();
}
}
