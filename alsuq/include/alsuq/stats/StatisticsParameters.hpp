#pragma once
#include "alsutils/parameters/Parameters.hpp"
#include "alsuq/types.hpp"
#include <boost/property_tree/ptree.hpp>
#include "alsuq/mpi/Configuration.hpp"
namespace alsuq {
namespace stats {

class StatisticsParameters : public alsutils::parameters::Parameters {
public:
    StatisticsParameters(const boost::property_tree::ptree& configuration);

    void setNumberOfSamples(size_t samples);
    size_t getNumberOfSamples() const;

    mpi::ConfigurationPtr getMpiConfiguration() const;
    void setMpiConfiguration(mpi::ConfigurationPtr value);

    void setPlatform(const std::string& platform);
    std::string getPlatform() const;
private:

    size_t samples =  0;

    mpi::ConfigurationPtr mpiConfiguration = nullptr;

    std::string platform = "cpu";
};
} // namespace stats
} // namespace alsuq
