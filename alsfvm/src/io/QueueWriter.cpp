#include "alsfvm/io/QueueWriter.hpp"

namespace alsfvm {
namespace io {

QueueWriter::QueueWriter(size_t queueLength,
    alsfvm::shared_ptr<volume::VolumeFactory>& volumeFactory)
    : queueSize(queueLength), nextIndex(0),
      volumeFactory(volumeFactory) {

}

void QueueWriter::write(const volume::Volume& conservedVariables,
    const volume::Volume& extraVariables,
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
