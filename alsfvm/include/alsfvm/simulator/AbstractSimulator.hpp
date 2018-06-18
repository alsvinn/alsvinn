/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include "alsfvm/types.hpp"
#include "alsfvm/io/Writer.hpp"
#include "alsfvm/integrator/TimestepAdjuster.hpp"

namespace alsfvm {
namespace simulator {

//! This is an abstract interface for the simulator class
//! for anyone wanting to use the uq component but with their own
//! simulator class.
//!
//! How to use:
//! \code{.cpp}
//! // save first timestep
//! simulator.callWriters();
//! while (!simulator.atEnd()) {
//!    simulator.performStep();
//! }
//! \endcode
class AbstractSimulator {
public:


    ///
    /// \return true if the simulation is finished, false otherwise.
    ///
    virtual bool atEnd() = 0;

    ///
    /// Performs one timestep
    ///
    virtual void performStep() = 0;

    ///
    /// Calls the writers.
    ///
    virtual void callWriters() = 0;

    ///
    /// \brief addWriter adds a writer, this will be called every time callWriter is called
    /// \param writer
    ///
    virtual void addWriter(alsfvm::shared_ptr<io::Writer> writer) = 0;


    ///
    /// \return the current simulation time.
    ///
    virtual real getCurrentTime() const = 0;

    ///
    /// \return the end time of the simulation.
    ///
    virtual real getEndTime() const = 0;


    //! Gets the current grid that is being used.
    virtual const std::shared_ptr<grid::Grid>& getGrid() const = 0;

    //! Gets the current grid that is being used.
    virtual std::shared_ptr<grid::Grid>& getGrid() = 0;

    //! Finalizes the computation, should be called at the end.
    virtual void finalize() = 0;

    //! Adds a timestep adjuster.
    //!
    //! The timestep adjuster is run as
    //! \code{.cpp}
    //! real newTimestep = someInitialValueFromCFL;
    //! for (auto adjuster : timestepAdjusters) {
    //!      newTimestep = adjuster(newTimestep);
    //! }
    //! \endcode
    //!
    //! the timestep adjuster is used to save at specific times.
    virtual void addTimestepAdjuster(
        alsfvm::shared_ptr<integrator::TimestepAdjuster>&
        adjuster) = 0;


};
} // namespace simulator
} // namespace alsfvm
