#pragma once
#include <map>
#include <string>
#include <cmath>
#include <boost/property_tree/ptree.hpp>

namespace alsutils {
namespace timer {

class TimerData {
public:
    TimerData() = default;

    template<class ...T>
    TimerData& getTimerData(const std::string& name, T... names) {
        return children[name].getTimerData(names...);
    }

    TimerData& getTimerData(const std::string& name);

    void addTime(double time);

    double getTotalTime() const;


    void print(const std::string& indent);

    boost::property_tree::ptree getTimesAsPropertyTree(double programTotalTime)
    const;
private:
    std::map<std::string, TimerData> children;
    double usedTime = 0;
    bool hasTimeData = false;
};


} // namespace timer
} // namespace alsutils
