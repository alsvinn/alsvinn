#include "alsutils/timer/TimerData.hpp"
#include <iostream>
namespace alsutils {
namespace timer {

TimerData& TimerData::getTimerData(const std::string& name) {
    return children[name];
}

void TimerData::addTime(double time) {
    usedTime += time;
    hasTimeData = true;
}

double TimerData::getTotalTime() const {
    if (hasTimeData) {
        return usedTime;
    }

    double totalTimeWithChildren = usedTime;

    for (const auto& child : children) {
        totalTimeWithChildren += child.second.getTotalTime();
    }

    return totalTimeWithChildren;
}

void TimerData::print(const std::string& indent) {
    std::cout << indent << "\"totalTime\" : " << getTotalTime() << ",\n";

    for (auto& child : children) {
        std::cout << indent << "\"" << child.first << "\" : " << "{\n";
        child.second.print(indent + "\t");
        std::cout << indent << "}\n";
    }
}

boost::property_tree::ptree TimerData::getTimesAsPropertyTree(
    double programTotalTime) const {

    boost::property_tree::ptree propertyTree;
    double totalTime = getTotalTime();
    propertyTree.put("totalTime", totalTime);
    propertyTree.put("percentOfProgramTotalTime",
        int(std::ceil(100 * totalTime / programTotalTime)));
    propertyTree.put("hasIndividualTimeData", hasTimeData);

    for (const auto& child : children) {
        propertyTree.add_child(child.first,
            child.second.getTimesAsPropertyTree(programTotalTime));
    }

    return propertyTree;
}

}
}
