#include "alsutils/get_username.hpp"
#include <boost/predef.h>
#include <vector>
#include <iostream>

#include <cstdlib>

#if BOOST_OS_LINUX
    #include <unistd.h>
#endif


std::string alsutils::getUsername() {
#if BOOST_OS_LINUX
    std::vector<char> buffer(32 * 1024, 0);

    auto returnValue = getlogin_r(buffer.data(), buffer.size());



    if (returnValue == 0) {
        return std::string(buffer.data());
    } else {
        auto usernameFromEnvironment = std::getenv("USER");

        if (usernameFromEnvironment) {
            return std::string(usernameFromEnvironment);
        }
    }

#endif

    return "Unknown user";
}
