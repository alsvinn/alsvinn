#include "alsuq/config/Setup.hpp"
#include "alsuq/generator/GeneratorFactory.hpp"
#include "alsuq/distribution/DistributionFactory.hpp"
#include "alsuq/mpi/SimpleLoadBalancer.hpp"
#include <boost/property_tree/xml_parser.hpp>
#include <fstream>
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


std::shared_ptr<run::Runner> Setup::makeRunner(const std::string &inputFilename, mpi::Config &mpiConfig)
{
    std::ifstream stream(inputFilename);
    ptree configuration;
    boost::property_tree::read_xml(stream, configuration);
    auto sampleGenerator = makeSampleGenerator(configuration);
    auto numberOfSamples = readNumberOfSamples(configuration);

    std::vector<size_t> samples;
    for (size_t i = 0; i < numberOfSamples; ++i) {
        samples.push_back(i);
    }

    auto simulatorCreator = std::make_shared<run::SimulatorCreator>(inputFilename, samples,
                                                                    mpiConfig.getCommunicator(), mpiConfig.getInfo());

    mpi::SimpleLoadBalancer loadBalancer(samples);

    auto samplesForProc = loadBalancer.getSamples(mpiConfig);

    auto runner = std::make_shared<run::Runner>(simulatorCreator, sampleGenerator, samplesForProc);

    return runner;
}

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

        auto distribution = distributionFactory.createDistribution(type,
                                                                   length,
                                                                   numberOfSamples,
                                                                   parametersToDistribution);


        auto generator = generatorFactory.makeGenerator(generatorName, length, numberOfSamples);
        generators[name] = std::make_pair(length, std::make_pair(generator, distribution));
    }

    return std::make_shared<samples::SampleGenerator> (generators);


}

size_t Setup::readNumberOfSamples(Setup::ptree &configuration)
{
    return configuration.get<real>("uq.samples");
}
}
}
