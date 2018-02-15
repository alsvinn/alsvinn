#pragma once
#include "alsuq/stats/StatisticsHelper.hpp"
#include "alsuq/types.hpp"
namespace alsuq {
namespace stats {

//! Computes the sturcture function given a direction
//!
//! Computes the structure function as
//!
//! \f[\sum_{i,j,k} \sum_{\tilde{i}=i-h}^{i+h}\cdot \sum_{\tildle{k}=k-h}^{k+h}|u_{(\tilde{i},\tilde{j},\tilde{i})}-u_{(i,j,k)}|^p/h^2\f]
class StructureCube : public StatisticsHelper {
public:
    StructureCube(const StatisticsParameters& parameters);


    //! Returns a list of ['structure_basic']
    virtual std::vector<std::string> getStatisticsNames() const override;




    virtual void computeStatistics(const alsfvm::volume::Volume& conservedVariables,
        const alsfvm::volume::Volume& extraVariables,
        const alsfvm::grid::Grid& grid,
        const alsfvm::simulator::TimestepInformation& timestepInformation) override;

    virtual void finalizeStatistics() override;



private:
    void computeStructure(alsfvm::volume::Volume& outputVolume,
        const alsfvm::volume::Volume& input);


    //! Helper function, computes the volume integral
    //!
    //! \note This must be called in order according to the ordering of h
    //! ie h=0 must be called first, then h=1, etc.
    void computeCube(alsfvm::memory::View<real>& output,
        const alsfvm::memory::View<const real>& input,
        int i, int j, int k, int h, int nx, int ny, int nz,
        int ngx, int ngy, int ngz, int dimensions);

    const real p;
    const int numberOfH;
    const std::string statisticsName;

};
} // namespace stats
} // namespace alsuq
