#pragma once
#include "alsuq/stats/Statistics.hpp"

namespace alsuq {
namespace stats {

//! Simple timer class
class StatisticsTimer : public Statistics {
public:
    StatisticsTimer(const std::string& name,
        std::shared_ptr<Statistics> statistics);

    ~StatisticsTimer();

    //! To be called when the statistics should be combined.
    virtual void combineStatistics();

    //! Adds a write for the given statistics name
    //! @param name the name of the statitics (one of the names returned in
    //!             getStatiticsNames()
    //! @param writer the writer to use
    virtual void addWriter(const std::string& name,
        std::shared_ptr<alsfvm::io::Writer>& writer);

    //! Returns a list of the names of the statistics being computed,
    //! typically this could be ['mean', 'variance']
    virtual std::vector<std::string> getStatisticsNames() const;


    virtual void computeStatistics(const alsfvm::volume::Volume& conservedVariables,
        const alsfvm::volume::Volume& extraVariables,
        const alsfvm::grid::Grid& grid,
        const alsfvm::simulator::TimestepInformation& timestepInformation);

    //! To be called in the end, this could be to eg compute the variance
    //! through M_2-mean^2 or any other postprocessing needed
    virtual void finalizeStatistics();

    virtual void writeStatistics(const alsfvm::grid::Grid& grids);

private:
    std::string name;
    const std::shared_ptr<Statistics> statistics;

    int statisticsTime = 0;
    int combineTime = 0;
    int finalizeTime = 0;
};
} // namespace stats
} // namespace alsuq
