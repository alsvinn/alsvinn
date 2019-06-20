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

#include "alsfvm/io/QueueWriter.hpp"

namespace alsfvm {
namespace io {

QueueWriter::QueueWriter(size_t queueLength,
    alsfvm::shared_ptr<volume::VolumeFactory>& volumeFactory)
    : queueSize(queueLength), nextIndex(0),
      volumeFactory(volumeFactory) {

}

void QueueWriter::write(const volume::Volume& conservedVariables,
    const grid::Grid& grid,
    const simulator::TimestepInformation& timestepInformation) {
    {
        std::unique_lock<std::mutex> scopedLock(mutex);

        while (waitingVolumes.size() == queueSize) {
            conditionVariable.wait(scopedLock, [&] () {
                return waitingVolumes.size() < allocatedVolumes.size();
            });
        }

        if (nextIndex < queueSize && nextIndex >= allocatedVolumes.size()) {
            size_t nx = conservedVariables.getNumberOfXCells();
            size_t ny = conservedVariables.getNumberOfYCells();
            size_t nz = conservedVariables.getNumberOfZCells();
            size_t ng = conservedVariables.getNumberOfXGhostCells();
            allocatedVolumes.push_back(volumeFactory->createConservedVolume(nx, ny, nz,
                    ng));
        }

        conservedVariables.copyTo(*allocatedVolumes[nextIndex]);
        waitingVolumes.push(allocatedVolumes[nextIndex]);
    }
    conditionVariable.notify_all();
}

void QueueWriter::pop(std::function<void(const volume::Volume&)> handler) {
    {
        std::unique_lock<std::mutex> scopedLock(mutex);

        while (waitingVolumes.empty()) {
            conditionVariable.wait(scopedLock, [&] () {
                return !waitingVolumes.empty();
            });
        }

        handler(*waitingVolumes.front());
        waitingVolumes.pop();
    }
    conditionVariable.notify_all();
}

}
}
