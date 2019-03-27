#pragma once
#include <vector>
#include <string>
#include <array>
namespace alsutils {
namespace debug {

struct BacktraceInformation {
    const std::string functionName = "";
    const std::string filename = "";
    const std::string lineNumber = "";

    BacktraceInformation(const std::string& functionName,
        const std::string& filename,
        const std::string lineNumber)
        : functionName(functionName), filename(filename), lineNumber(lineNumber) {

    }
};

//! Gets the stacktrace on GCC based systems (current not supported
//! on other compilers). Uses the following functions in gcc
//!
//!     http://www.gnu.org/software/libc/manual/html_node/Backtraces.html
//!
//! @return a vector where position 0 is the name of the first function called,
//!                  position 1 is the name of the second function, and so on
std::vector<BacktraceInformation> getStacktrace();

//! Gets the stacktrace on GCC based systems (current not supported
//! on other compilers). Uses the following functions in gcc
//!
//!     http://www.gnu.org/software/libc/manual/html_node/Backtraces.html
//!
//! @return a string on the form toplevelfunctionname.somefunction.<...>.lastfunctionname
std::string getShortStacktrace();

//! Gets the stacktrace on GCC based systems (current not supported
//! on other compilers). Uses the following functions in gcc
//!
//!     http://www.gnu.org/software/libc/manual/html_node/Backtraces.html
//!
//! @return a string on the form toplevelfunctionname\nsomefunction\n<...>\nlastfunctionname
std::string getLongStacktrace();

}
}
