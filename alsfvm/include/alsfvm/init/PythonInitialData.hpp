#pragma once
#include "alsfvm/init/InitialData.hpp"
#include "alsfvm/equation/CellComputer.hpp"
#include "alsfvm/init/Parameters.hpp"
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
        ///  # coordinates are stored in the variables x, y and z
        ///  rho = ...
        ///  ux = ...
        ///  uy = ...
        ///  uz = ...
        ///  p = ...
        /// \endcode
        ///
        ///
        /// The momentum (m) and energy will be computed automatically.
        ///
        PythonInitialData(const std::string& programString,
            const Parameters& parameters);


        ///
        /// \brief setInitialData sets the initial data
        /// \param conservedVolume conserved volume to fill
        /// \param extraVolume the extra volume
        /// \param cellComputer an instance of the cell computer for the equation
        /// \param primitiveVolume an instance of the primtive volume for the equation
        /// \param grid underlying grid.
        /// \note All volumes need to have the correct size. All volumes will at the
        /// end be written to.
        /// \note This is not an efficient implementation, so it should really only
        /// be used for initial data!
        ///
        virtual void setInitialData(volume::Volume& conservedVolume,
                            volume::Volume& extraVolume,
                            volume::Volume& primitiveVolume,
                            equation::CellComputer& cellComputer,
                            grid::Grid& grid);

        virtual void setParameters(const Parameters& parameters);

    private:
        Parameters parameters;
        std::string programString;

    };
} // namespace alsfvm
} // namespace init
