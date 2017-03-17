#pragma once
#include "alsuq/types.hpp"
#include "alsuq/stats/Statistics.hpp"
#include "alsuq/stats/StatisticsSnapshot.hpp"

namespace alsuq { namespace stats { 

class StatisticsHelper : public Statistics {
public:
    //! Add a writer to write the statistics to file
    //!
    //! @param writer the writer to add
    void addWriter(const std::string& name,
                   std::shared_ptr<alsfvm::io::Writer>& writer);

    //! Should be called at the end of the simulation
    virtual void combineStatistics();



    //! Writes the statistics to file
    virtual void writeStatistics(const alsfvm::grid::Grid &grid);



protected:
    std::map<real, std::map<std::string, StatisticsSnapshot> > snapshots;

    //! Utility function.
    //!
    //! If the given timstep is already created, return that timestep,
    //! otherwise creates a new snapshot
    //!
    //! \note Uses the size of the given volume
    StatisticsSnapshot& findOrCreateSnapshot(const std::string& name,
                                             const alsfvm::simulator::TimestepInformation& timestepInformation,
                                             const alsfvm::volume::Volume& conservedVariables,
                                             const alsfvm::volume::Volume& extraVariables);

    //! Utility function.
    //!
    //! If the given timstep is already created, return that timestep,
    //! otherwise creates a new snapshot
    StatisticsSnapshot& findOrCreateSnapshot(const std::string& name,
                                             const alsfvm::simulator::TimestepInformation& timestepInformation,
                                             const alsfvm::volume::Volume& conservedVariables,
                                             const alsfvm::volume::Volume& extraVariables,
                                             size_t nx, size_t ny, size_t nz);
private:
    std::map<std::string, std::vector<std::shared_ptr<alsfvm::io::Writer> > > writers;

    alsuq::mpi::Config mpiConfig;
};
} // namespace stats
                } // namespace alsuq
