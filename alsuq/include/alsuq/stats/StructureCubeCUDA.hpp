#pragma once
#include "alsuq/stats/StatisticsHelper.hpp"
#include "alsuq/types.hpp"
#include <thrust/device_vector.h>

namespace alsuq {
namespace stats {

//! Computes the sturcture function given a direction
//!
//! Computes the structure function as
//!
//! \f[\sum_{i,j,k} \sum_{\tilde{i}=i-h}^{i+h}\cdot \sum_{\tildle{k}=k-h}^{k+h}|u_{(\tilde{i},\tilde{j},\tilde{i})}-u_{(i,j,k)}|^p/h^2\f]
class StructureCubeCUDA : public StatisticsHelper {
    public:
        StructureCubeCUDA(const StatisticsParameters& parameters);


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

        const real p;
        const int numberOfH;
        const std::string statisticsName;


        // For now we use thurst's version of reduce, and to make everything
        // play nice, we use thrust device vector to hold the temporary results.
        thrust::device_vector<real> structureOutput;
};
} // namespace stats
} // namespace alsuq
