#include "alsuq/stats/StatisticsHelper.hpp"
#include "alsuq/types.hpp"
#include <thrust/device_vector.h>

namespace alsuq {
namespace stats {

//! Computes the sturcture function given a direction
//!
//! Ie in parameters it gets an in direction that corresponds to the unit
//! direction vectors \f$e\f$
//!
//! It then computes the structure function as
//!
//! \f[\sum_{i,j,k} |u_{(i,j,k) + e}-u_{(i,j,k)}|^p\f]
class StructureBasicCUDA : public StatisticsHelper {
    public:
        StructureBasicCUDA(const StatisticsParameters& parameters);


        //! Returns a list of ['structure_basic']
        virtual std::vector<std::string> getStatisticsNames() const;




        virtual void computeStatistics(const alsfvm::volume::Volume& conservedVariables,
            const alsfvm::volume::Volume& extraVariables,
            const alsfvm::grid::Grid& grid,
            const alsfvm::simulator::TimestepInformation& timestepInformation) override;

        virtual void finalize() override;



    private:
        void computeStructure(alsfvm::volume::Volume& outputVolume,
            const alsfvm::volume::Volume& input);

        const size_t direction;
        const real p;
        const ivec3 directionVector;
        const size_t numberOfH;
        const std::string statisticsName;

        // For now we use thurst's version of reduce, and to make everything
        // play nice, we use thrust device vector to hold the temporary results.
        thrust::device_vector<real> structureOutput;

};
} // namespace stats
} // namespace alsuq
