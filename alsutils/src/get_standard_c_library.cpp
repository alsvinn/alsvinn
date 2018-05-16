#include "alsutils/get_standard_c_library.hpp"
#include <boost/predef.h>
#if BOOST_OS_LINUX
#include <gnu/libc-version.h>
#endif

namespace alsutils {
std::string getStandardCLibrary() {
#if BOOST_OS_LINUX
    return std::string("GNU libc ") + std::string(gnu_get_libc_version())
            + std::string(" ") + std::string(gnu_get_libc_release());
#else
    return "Unknown standard C library";
#endif
}
}
