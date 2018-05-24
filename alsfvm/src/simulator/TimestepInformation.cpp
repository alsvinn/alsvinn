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

#include "alsfvm/simulator/TimestepInformation.hpp"

namespace alsfvm {
namespace simulator {

TimestepInformation::TimestepInformation(real currentTime,
    size_t numberOfStepsPerformed)
    : currentTime(currentTime), numberOfStepsPerformed(numberOfStepsPerformed) {

}

TimestepInformation::TimestepInformation()
    : currentTime(0), numberOfStepsPerformed(0) {

}

void TimestepInformation::incrementTime(real dt) {
    currentTime += dt;
    numberOfStepsPerformed++;
}

real TimestepInformation::getCurrentTime() const {
    return currentTime;
}

size_t TimestepInformation::getNumberOfStepsPerformed() const {
    return numberOfStepsPerformed;
}

} // namespace simulator
} // namespace alsfvm

