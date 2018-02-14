#include "alsfvm/io/io_utils.hpp"
#include <sstream>

namespace alsfvm {
namespace io {


///
/// \brief getOutputname creates the output filename
/// \note This does not include an extension
/// \param filename the base filename (eg. "simulation")
/// \param snapshotNumber the current snapshot number (this not the timestep
///        number)
/// \note snapshotNumber is essentially the number of snapshots that has been
///                      saved up until now.
/// \return the filename with extra timestep information attached.
///
std::string getOutputname(const std::string& filename,
    size_t snapshotNumber) {
    std::stringstream ss;
    ss << filename << "_" << snapshotNumber;

    return ss.str();
}
}
}
