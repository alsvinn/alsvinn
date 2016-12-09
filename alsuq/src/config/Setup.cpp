#include "alsuq/config/Setup.hpp"
#include "alsuq/generator/GeneratorFactory.hpp"
#include "alsuq/distribution/DistributionFactory.hpp"

namespace alsuq { namespace config {
// example:
// <samples>1024</samples>
// <generator>auto</generator>
// <parameters>
//   <parameter>
//     <name>a</name>
//     <length>40</length>
//     <type>uniform</type>
//   </parameter>
// </parameters>
// <stats>
//   <stat>
//     meanvar
//   </stat>
// </stats>


std::shared_ptr<samples::SampleGenerator> Setup::makeSampleGenerator(Setup::ptree& configuration)
{
    auto numberOfSamples = readNumberOfSamples(configuration);

    samples::SampleGenerator::GeneratorDistributionMap generators;

    auto generatorName = configuration.get<std::string>("uq.generator");

    if (generatorName=="auto") {
        generatorName = "well512a";
    }

    auto parametersNode = configuration.get_child("uq.parameters");
    generator::GeneratorFactory generatorFactory;
    distribution::DistributionFactory distributionFactory;
    for (auto parameterNode : parametersNode) {
        auto name = parameterNode.second.get<std::string>("name");
        auto length = parameterNode.second.get<size_t>("length");
        auto type = parameterNode.second.get<std::string>("type");

        distribution::Parameters parametersToDistribution;
        parametersToDistribution.setParameter("lower", 0);
        parametersToDistribution.setParameter("upper", 1);

        parametersToDistribution.setParameter("mean", 0);
        parametersToDistribution.setParameter("sd", 1);

        auto distribution = distributionFactory.createDistribution(name,
                                                                   length,
                                                                   numberOfSamples,
                                                                   parametersToDistribution);


        auto generator = generatorFactory.makeGenerator(generatorName, length, numberOfSamples);
        generators[name] = std::make_pair(numberOfSamples, std::make_pair(generator, distribution));
    }

    return std::make_shared<samples::SampleGenerator> (generators);


}

size_t Setup::readNumberOfSamples(Setup::ptree &configuration)
{
    return configuration.get<real>("uq.samples");
}
}
}
