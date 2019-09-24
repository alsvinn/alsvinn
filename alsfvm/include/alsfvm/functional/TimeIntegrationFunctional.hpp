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
#include "alsfvm/io/Writer.hpp"
#include "alsfvm/functional/Functional.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/memory/MemoryFactory.hpp"

namespace alsfvm {
namespace functional {

//! This lets you time integrate a functional, that is, for a functional
//! g (interpreted in the loose sense), this will compute
//!
//! \f[\int_{t-\tau}^{t+\tau} g(u(t))\; dt\f]
//!
//! @note this computes the time integral *without* averaging,
//!       if you want to get the time averaged quantity, you have to divide
//!       the output by \f$2\tau\f$.
//!
//! @note It is not really easy to combine this into the time integration class
//!       for writing. The reason for this is that we only selectively want to call
//!       the functional, to minimize computational work.
class TimeIntegrationFunctional : public io::Writer {
public:

    TimeIntegrationFunctional(volume::VolumeFactory volumeFactory,
        io::WriterPointer writer,
        FunctionalPointer functional,
        double time,
        double timeRadius);

    virtual void write(const volume::Volume& conservedVariables,
        const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation) override;

    virtual void finalize(const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation) override;

private:
    void makeVolumes(const grid::Grid& grid, const volume::Volume& volume);
    volume::VolumeFactory volumeFactory;
    io::WriterPointer writer;
    FunctionalPointer functional;

    volume::VolumePointer conservedVolume;

    const double time;
    const double timeRadius;
    double lastTime = 0;

    ivec3 functionalSize;

};
} // namespace functional
} // namespace alsfvm
