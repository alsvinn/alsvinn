#pragma once
#include "alsfvm/volume/Volume.hpp"

namespace alsfvm { namespace equation { 

///
/// \brief The CellComputer class defines some useful per cell computations
///
    class CellComputer {
    public:
        ///
        /// \brief computeExtraVariables computes the extra variables (eg. pressure for euler)
        /// \param[in] conservedVariables the conserved variables to read from
        /// \param[out] extraVariables the extra variables to write to
        ///
        virtual void computeExtraVariables(const volume::Volume& conservedVariables,
                                           volume::Volume& extraVariables) = 0;

		///
		/// Computes the maximum wavespeed across all direction
		/// \param conservedVariables the conserved variables (density, momentum, Energy for Euler)
		/// \param extraVariables the extra variables (pressure and velocity for Euler)
		/// \return the maximum wave speed (absolute value)
		///
		virtual real computeMaxWaveSpeed(const volume::Volume& conservedVariables,
			const volume::Volume& extraVariables) = 0;

		/// 
		/// Checks if all the constraints for the equation are met
		///	\param conservedVariables the conserved variables (density, momentum, Energy for Euler)
		/// \param extraVariables the extra variables (pressure and velocity for Euler)
		/// \return true if it obeys the constraints, false otherwise
		///
		virtual bool obeysConstraints(const volume::Volume& conservedVariables,
			const volume::Volume& extraVariables) = 0;
    };
} // namespace alsfvm
} // namespace equation
