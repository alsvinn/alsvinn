#pragma once
#include "alsfvm/init/InitialData.hpp"

namespace alsfvm { namespace init { 

///
/// \brief The PythonInitialData class sets the initial data through
/// a python string.
///
    class PythonInitialData : public InitialData {
    public:
        ///
        /// \brief PythonInitialData constructs the object
        /// \param programString the string containing the full python program.
        ///
        /// The programString should be in the following format:
        /// \code{.py}
        /// def initialData(x, y, z, resultValue):
        ///     resultValue.rho = ...
        ///     resultValue.u = ...
        ///     resultValue.p = ...
        /// \endcode
        ///
        /// The momentum (m) and energy will be computed automatically.
        ///
        PythonInitialData(const std::string& programString);

        ///
        /// \brief setInitialData sets the initial data
        /// \param conservedVolume conserved volume to fill
        /// \param extraVolume the extra volume
        /// \param grid underlying grid.
        ///
        virtual void setInitialData(volume::Volume& conservedVolume,
                            volume::Volume& extraVolume,
                            grid::Grid& grid);

    };
} // namespace alsfvm
} // namespace init
