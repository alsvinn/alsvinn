#include "alsutils/get_hostname.hpp"
#include <boost/predef.h>
#include <vector>

#if BOOST_OS_LINUX
    #include <unistd.h>
#endif
namespace alsutils {
std::string getHostname() {
#if BOOST_OS_LINUX
    std::vector<char> buffer(32 * 1024, 0);

    auto returnValue = gethostname(buffer.data(), buffer.size());

    if (returnValue == 0) {
        return std::string(buffer.data());
    }

#endif
    return "Unknown hostname";
}
}
