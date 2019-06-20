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
#include "alsfvm/io/FixedIntervalWriter.hpp"
#include "alsfvm/functional/Functional.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/memory/MemoryFactory.hpp"

namespace alsfvm {
namespace functional {


///
/// \brief The IntervalFunctionalWriter class is a decorator for another writer.
/// Its purpose is to only call the underlying Writer object at fixed time intervals.
///
/// This class is useful if you only want to save every x seconds of simulation. This class assume you
/// already decorates it with the alsfvm::io::FixedIntervalWriter
///
class IntervalFunctionalWriter : public io::Writer {
public:

    IntervalFunctionalWriter(volume::VolumeFactory volumeFactory,
        io::WriterPointer writer,
        FunctionalPointer functional
    );

    virtual void write(const volume::Volume& conservedVariables,
        const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation) override;


private:
    void makeVolumes(const grid::Grid& grid);
    volume::VolumeFactory volumeFactory;
    io::WriterPointer writer;
    FunctionalPointer functional;

    volume::VolumePointer conservedVolume;



    ivec3 functionalSize;

};
} // namespace functional
} // namespace alsfvm
