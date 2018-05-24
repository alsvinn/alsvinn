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
#include "alsfvm/simulator/TimestepInformation.hpp"
#include "alsfvm/volume/VolumePair.hpp"

namespace alsuq {
namespace stats {

//! Holds a snapshot (in time) of the current statistics.
class StatisticsSnapshot {
public:
    StatisticsSnapshot() {}
    StatisticsSnapshot(alsfvm::simulator::TimestepInformation timestepInformation,
        alsfvm::volume::VolumePair volumes);

    alsfvm::simulator::TimestepInformation& getTimestepInformation();

    alsfvm::volume::VolumePair& getVolumes();

private:
    alsfvm::simulator::TimestepInformation timestepInformation;
    alsfvm::volume::VolumePair volumes;
};
} // namespace stats
} // namespace alsuq
