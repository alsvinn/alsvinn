#include "alsutils/get_os_name.hpp"
#include <boost/predef.h>
#include <sstream>

#if BOOST_OS_LINUX
#include <sys/utsname.h>
#endif
namespace alsutils {
std::string getOSName() {
#if BOOST_OS_LINUX
    utsname buffer;
    uname(&buffer);

    std::stringstream name;

    name <<"Linux: "<< buffer.sysname << " " << buffer.release << " " << buffer.version << " "<<  buffer.machine;

    return name.str();
#elif BOOST_OS_MACOS
    return BOOST_OS_MACOS_NAME;
#elif BOOST_OS_WINDOWS
    return BOOST_OS_WINDOWS_NAME;
#else
    return "Unknown operating system";
#endif
}
}
