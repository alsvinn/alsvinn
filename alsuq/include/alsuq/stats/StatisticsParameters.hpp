#pragma once
#include "alsuq/types.hpp"
#include <boost/property_tree/ptree.hpp>
#include "alsuq/mpi/Configuration.hpp"
namespace alsuq {
namespace stats {

class StatisticsParameters {
public:


    void setNumberOfSamples(size_t samples);
    size_t getNumberOfSamples() const;

    const std::string getParameterAsString(const std::string& name) const;


    real getParameterAsDouble(const std::string& name) const;


    int getParameterAsInteger(const std::string& name) const;
    void setConfiguration(const boost::property_tree::ptree& configuration);
    mpi::ConfigurationPtr getMpiConfiguration() const;
    void setMpiConfiguration(mpi::ConfigurationPtr value);

private:
    size_t samples;
    boost::property_tree::ptree configuration;
    mpi::ConfigurationPtr mpiConfiguration;
};
} // namespace stats
} // namespace alsuq
