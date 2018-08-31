#include "alsutils/timer/TimerDatabase.hpp"
#include <iostream>

namespace alsutils {
namespace timer {

TimerDatabase& TimerDatabase::getInstance() {
    static TimerDatabase instance;
    return instance;
}

void TimerDatabase::print() {
    std::cout << "{\n";
    root.print("\t");
    std::cout << "}\n";
}

boost::property_tree::ptree TimerDatabase::getTimesAsPropertyTree() const {
    boost::property_tree::ptree propertyTree;

    propertyTree.add_child("allTimedEvents",
        root.getTimesAsPropertyTree(root.getTotalTime()));

    return propertyTree;
}

}
}
