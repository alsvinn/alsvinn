#include "alsutils/mpi/safe_call.hpp"
#ifdef ALSVINN_USE_MPI
#include <mpi.h>
#include <string>
#include <vector>
#endif


namespace alsutils {
namespace mpi {
std::string getMPIVersion() {
#ifdef ALSVINN_USE_MPI
    try {
        std::vector<char> version(MPI_MAX_LIBRARY_VERSION_STRING+1);
        int length;
        MPI_SAFE_CALL(MPI_Get_library_version(version.data(), &length));

        return std::string(version.begin(), version.begin() + length - 1);
    } catch (...) {
        return "Unknown MPI version";
    }
#else
    return "Unknown MPI version";
#endif
}
}
}
