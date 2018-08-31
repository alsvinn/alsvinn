#pragma once
#include "alsutils/timer/TimerData.hpp"
#include <boost/property_tree/ptree.hpp>
namespace alsutils {
namespace timer {

class TimerDatabase {
    TimerDatabase() {}
public:
    static TimerDatabase& getInstance();

    template<class ...T>
    TimerData& getTimerData(T... names) {
        return root.getTimerData(names...);
    }


    void print();

    boost::property_tree::ptree getTimesAsPropertyTree() const;

private:
    TimerData root;
};
} // namespace timer
} // namespace alsutils
