#include "alsuq/stats/StatisticsFactory.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsuq/stats/StatisticsTimer.hpp"

namespace alsuq {
namespace stats {

namespace {
class StatististicsList {


    public:
        static StatististicsList& instance() {
            static StatististicsList list;

            return list;
        }
        std::map<std::string, std::map<std::string, StatisticsFactory::StatisticsCreator> >
        creators;
    private:
        StatististicsList() {}

};
}

void StatisticsFactory::registerStatistics(const std::string& platform,
    const std::string& name,
    StatisticsFactory::StatisticsCreator maker) {
    auto& list = StatististicsList::instance().creators;

    if (list[platform].find(name) != list[platform].end()) {
        THROW("'" << name << "' already registered as a Statistic");
    }

    list[platform][name] = maker;
}

StatisticsFactory::StatisticsPointer StatisticsFactory::makeStatistics(
    const std::string& platform,
    const std::string& name,
    const StatisticsParameters& params) {
    auto& list = StatististicsList::instance().creators;

    if (list[platform].find(name) == list[platform].end()) {
        THROW("Unknown statistics: " << name);
    }

    StatisticsPointer pointer;
    pointer.reset(new StatisticsTimer(name, list[platform][name](params)));

    return pointer;
}

}
}
