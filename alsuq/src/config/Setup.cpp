#include "alsuq/config/Setup.hpp"
#include "alsuq/generator/GeneratorFactory.hpp"
#include "alsuq/distribution/DistributionFactory.hpp"
#include "alsuq/mpi/SimpleLoadBalancer.hpp"
#include <boost/property_tree/xml_parser.hpp>
#include <fstream>
#include "alsuq/stats/StatisticsFactory.hpp"
#include "alsfvm/io/WriterFactory.hpp"
#include "alsuq/stats/FixedIntervalStatistics.hpp"
#include <boost/algorithm/string.hpp>
#include "alsutils/log.hpp"

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


std::shared_ptr<run::Runner> Setup::makeRunner(const std::string &inputFilename,
                                               alsutils::mpi::ConfigurationPtr mpiConfigurationWorld,
                                               int multiSample, ivec3 multiSpatial)
{
    std::ifstream stream(inputFilename);
    ptree configurationBase;
    boost::property_tree::read_xml(stream, configurationBase);
    auto configuration = configurationBase.get_child("config");
    auto sampleGenerator = makeSampleGenerator(configuration);
    auto numberOfSamples = readNumberOfSamples(configuration);

    std::vector<size_t> samples;
    samples.reserve(numberOfSamples);
    for (size_t i = 0; i < numberOfSamples; ++i) {
        samples.push_back(i);
    }


    mpi::SimpleLoadBalancer loadBalancer(samples);

    auto loadBalanceConfiguration = loadBalancer.loadBalance(multiSample,
                                                             multiSpatial,
                                                             *mpiConfigurationWorld);
    auto& samplesForProc = std::get<0>(loadBalanceConfiguration);
    auto statisticalConfiguration = std::get<1>(loadBalanceConfiguration);
    auto spatialConfiguration = std::get<2>(loadBalanceConfiguration);


    auto simulatorCreator = std::make_shared<run::SimulatorCreator>(inputFilename,
                                                                    spatialConfiguration,
                                                                    statisticalConfiguration,
                                                                    mpiConfigurationWorld,
                                                                    multiSpatial);


    auto runner = std::make_shared<run::Runner>(simulatorCreator, sampleGenerator, samplesForProc,
                                                statisticalConfiguration);
    auto statistics  = createStatistics(configuration, statisticalConfiguration);
    runner->setStatistics(statistics);
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

        parametersToDistribution.setParameter("a", 0);
        parametersToDistribution.setParameter("b", 1);

	
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


// <stats>
//
//   <stat>
//   <name>
//   structure
//   </name>
//   <numberOfSaves>1</numberOfSaves>
//   <writer>
//
//     <type>netcdf</type>
//     <basename>kh_structure</basename>
//   </writer>
//   </stat>
//   <stat>
//   <name>
//     meanvar
//   </name>
//   <numberOfSaves>1</numberOfSaves>
//   <writer>
//     <type>netcdf</type>
//     <basename>kh_structure</basename>
//   </writer>
//   </stat>
// </stats>
std::vector<std::shared_ptr<stats::Statistics> > Setup::createStatistics(Setup::ptree &configuration,
                                                                         mpi::ConfigurationPtr statisticalConfiguration)
{
    auto statisticsNodes = configuration.get_child("uq.stats");
    stats::StatisticsFactory statisticsFactory;
    alsfvm::io::WriterFactory writerFactory;
    auto platform = configuration.get<std::string>("fvm.platform");
    std::vector<std::shared_ptr<stats::Statistics> > statisticsVector;
    for (auto& statisticsNode : statisticsNodes) {
        auto name = statisticsNode.second.get<std::string>("name");
        boost::trim(name);
        stats::StatisticsParameters parameters;
        parameters.setMpiConfiguration(statisticalConfiguration);
        parameters.setNumberOfSamples(readNumberOfSamples(configuration));
        parameters.setConfiguration(statisticsNode.second);
        auto statistics = statisticsFactory.makeStatistics(platform, name, parameters);
        auto numberOfSaves = statisticsNode.second.get<size_t>("numberOfSaves");

        ALSVINN_LOG(INFO, "statistics.numberOfSaves = " << numberOfSaves);

        // Make writer
        std::string type = statisticsNode.second.get<std::string>("writer.type");
        std::string basename = statisticsNode.second.get<std::string>("writer.basename");
        for (auto statisticsName : statistics->getStatisticsNames()) {

            auto outputname = basename + "_" + statisticsName;
            auto baseWriter = writerFactory.createWriter(type, outputname);
            statistics->addWriter(statisticsName, baseWriter);
        }
        real endTime = configuration.get<real>("fvm.endTime");
        real timeInterval = endTime / numberOfSaves;
        auto statisticsInterval =
                std::shared_ptr<stats::Statistics>(
                    new stats::FixedIntervalStatistics(statistics, timeInterval,
                                                       endTime));
        statisticsVector.push_back(statisticsInterval);

    }

    return statisticsVector;
}

size_t Setup::readNumberOfSamples(Setup::ptree &configuration)
{
    return configuration.get<real>("uq.samples");
}
}
}
