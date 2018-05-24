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

namespace alsfvm {
namespace integrator {

//! Lets observers adjust the wavespeed (this is used for eg. MPI to take the maximum
//! over all cells)
class WaveSpeedAdjuster {
public:
    virtual ~WaveSpeedAdjuster() {};
    virtual real adjustWaveSpeed(real waveSpeed)  = 0;
};

typedef alsfvm::shared_ptr<WaveSpeedAdjuster> WaveSpeedAdjusterPtr;
} // namespace integrator
} // namespace alsfvm
