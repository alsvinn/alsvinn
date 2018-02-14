#include "alsuq/stats/StatisticsSnapshot.hpp"

namespace alsuq {
namespace stats {

StatisticsSnapshot::StatisticsSnapshot(
    alsfvm::simulator::TimestepInformation timestepInformation,
    alsfvm::volume::VolumePair volumes)
    : timestepInformation(timestepInformation),
      volumes(volumes) {

}

alsfvm::simulator::TimestepInformation&
StatisticsSnapshot::getTimestepInformation() {
    return timestepInformation;
}

alsfvm::volume::VolumePair& StatisticsSnapshot::getVolumes() {
    return volumes;
}

}
}
