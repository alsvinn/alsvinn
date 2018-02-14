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
