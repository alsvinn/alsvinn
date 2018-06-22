#pragma once
#include <boost/python.hpp>
#include "alsutils/error/Exception.hpp"
namespace alsfvm {
namespace python {

//! Small macro to handle python exceptions, not in a function to allow for nicer
//! stack trace.
//!
//! From https://stackoverflow.com/a/6576177
#define HANDLE_PYTHON_EXCEPTION \
    std::string pythonErrorMessage = "none"; \
    if (PyErr_Occurred()) { \
                pythonErrorMessage = alsfvm::python::handle_pyerror();  \
            } \
            boost::python::handle_exception(); \
            PyErr_Clear(); \
            THROW("Error running python script:\n" << pythonErrorMessage)



//! decode a Python exception into a string
//!
//! This code is from https://stackoverflow.com/a/6576177
inline std::string handle_pyerror() {
    using namespace boost::python;
    using namespace boost;

    PyObject* exc, *val, *tb;
    object formatted_list, formatted;
    PyErr_Fetch(&exc, &val, &tb);
    handle<> hexc(exc), hval(allow_null(val)), htb(allow_null(tb));
    object traceback(import("traceback"));

    if (!tb) {
        object format_exception_only(traceback.attr("format_exception_only"));
        formatted_list = format_exception_only(hexc, hval);
    } else {
        object format_exception(traceback.attr("format_exception"));
        formatted_list = format_exception(hexc, hval, htb);
    }

    formatted = str("\n").join(formatted_list);
    return extract<std::string>(formatted);
}
}
}
