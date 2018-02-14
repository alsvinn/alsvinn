#pragma once
#include "alsfvm/io/Writer.hpp"
#include "alsuq/mpi/Configuration.hpp"

namespace alsuq {
namespace stats {

//! Abstract base class for computing statistics (mean, variance, structure
//! functions, etc)
class Statistics : public alsfvm::io::Writer {
    public:
        virtual ~Statistics() {}

        ///
        /// Passes the information onto computeStats
        ///
        virtual void write(const alsfvm::volume::Volume& conservedVariables,
            const alsfvm::volume::Volume& extraVariables,
            const alsfvm::grid::Grid& grid,
            const alsfvm::simulator::TimestepInformation& timestepInformation);


        //! To be called when the statistics should be combined.
        virtual void combineStatistics() = 0;

        //! Adds a write for the given statistics name
        //! @param name the name of the statitics (one of the names returned in
        //!             getStatiticsNames()
        //! @param writer the writer to use
        virtual void addWriter(const std::string& name,
            std::shared_ptr<alsfvm::io::Writer>& writer) = 0;

        //! Returns a list of the names of the statistics being computed,
        //! typically this could be ['mean', 'variance']
        virtual std::vector<std::string> getStatisticsNames() const = 0;


        virtual void computeStatistics(const alsfvm::volume::Volume& conservedVariables,
            const alsfvm::volume::Volume& extraVariables,
            const alsfvm::grid::Grid& grid,
            const alsfvm::simulator::TimestepInformation& timestepInformation) = 0;

        //! To be called in the end, this could be to eg compute the variance
        //! through M_2-mean^2 or any other postprocessing needed
        virtual void finalize() = 0;

        virtual void writeStatistics(const alsfvm::grid::Grid& grid) = 0;

};
} // namespace stats
} // namespace alsuq
