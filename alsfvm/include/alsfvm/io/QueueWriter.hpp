#pragma once
#include "alsfvm/io/Writer.hpp"
#include <mutex>
#include <vector>
#include <queue>
#include <condition_variable>
#include "alsfvm/volume/VolumeFactory.hpp"
#include <functional>

namespace alsfvm {
namespace io {

///
/// \brief The QueueWriter class writes the data to a blocking queue.
///
/// This is ideal if you have two simulations that need to communicate with
/// eachother.
///
class QueueWriter : public Writer {
public:

    ///
    /// \param queueLength the number of elements to hold in the queue.
    /// \param volumeFactory the volume factory to use to create new volumes.
    ///
    /// \note queueLength must be larger than 0
    ///
    QueueWriter(size_t queueLength,
        alsfvm::shared_ptr<volume::VolumeFactory>& volumeFactory);

    ///
    /// \brief write writes the data to the queue
    /// \param conservedVariables the conservedVariables to write
    /// \param extraVariables the extra variables to write
    /// \param grid the grid that is used (describes the _whole_ domain)
    /// \param timestepInformation
    ///
    virtual void write(const volume::Volume& conservedVariables,
        const volume::Volume& extraVariables,
        const grid::Grid& grid,
        const simulator::TimestepInformation& timestepInformation);

    void pop(std::function<void(const volume::Volume&)> handler);
private:
    size_t queueSize;
    size_t nextIndex = 0;
    // Allocated volumes, these may be in use (check refcount)
    std::vector<alsfvm::shared_ptr<volume::Volume> > allocatedVolumes;

    // These are the volumes that have not been used yet.
    std::queue<alsfvm::shared_ptr<volume::Volume> > waitingVolumes;

    alsfvm::shared_ptr<volume::VolumeFactory> volumeFactory;

    std::mutex mutex;

    std::condition_variable conditionVariable;
};
} // namespace alsfvm
} // namespace io
