#pragma once
#include <string>
namespace alsutils {
//! Returns a descriptive name of the standard C library being used
//! @note only works on Linux so far, otherwise returns "Unknown"
std::string getStandardCLibrary();
}
