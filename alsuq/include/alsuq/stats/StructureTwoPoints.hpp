#pragma once
#include "alsuq/stats/StatisticsHelper.hpp"
#include "alsuq/types.hpp"
namespace alsuq {
namespace stats {

//! Computes the sturcture function given a direction
//!
//! Ie in parameters it gets an in direction that corresponds to the unit
//! direction vectors \f$e_i\f$.
//!
//! It then computes the structure function as
//!
//! \f[\sum_{i,j,k} (u_{(i,j,k) + e_1}-u_{(i,j,k)})(u_{(i,j,k) + e_2}-u_{(i,j,k))^2\f]
class StructureTwoPoints : public StatisticsHelper {
    public:
        StructureTwoPoints(const StatisticsParameters& parameters);


        //! Returns a list of ['structure_2pt']
        virtual std::vector<std::string> getStatisticsNames() const;




        virtual void computeStatistics(const alsfvm::volume::Volume& conservedVariables,
            const alsfvm::volume::Volume& extraVariables,
            const alsfvm::grid::Grid& grid,
            const alsfvm::simulator::TimestepInformation& timestepInformation) override;

        virtual void finalize() override;



    private:
        void computeStructure(alsfvm::volume::Volume& outputVolume,
            const alsfvm::volume::Volume& input);

        const size_t direction1;
        const size_t direction2;
        const ivec3 directionVector1;
        const ivec3 directionVector2;
        const size_t numberOfH;
        const std::string statisticsName;

};
} // namespace stats
} // namespace alsuq
