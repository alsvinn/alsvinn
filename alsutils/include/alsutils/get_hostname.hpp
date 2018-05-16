#pragma once
#include <string>
namespace alsutils {

//! Tries in a portable way to get the hostname of the current computer
//! This will in most instances call gethostname
//! See http://man7.org/linux/man-pages/man2/gethostname.2.html
//! and https://msdn.microsoft.com/library/windows/desktop/ms738527(v=vs.85).aspx/
std::string getHostname();
}
