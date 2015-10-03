#pragma once
#include <iostream>
#include "alsfvm/simulator/Simulator.hpp"
#include <memory>
#include <boost/property_tree/ptree.hpp>

namespace alsfvm { namespace config { 

    class SimulatorSetup {
    public:
        typedef boost::property_tree::ptree ptree;
        boost::shared_ptr<simulator::Simulator>
            readSetupFromFile(const std::string& filename);

    protected:
        boost::shared_ptr<grid::Grid> createGrid(const ptree& configuration);
        real readEndTime(const ptree& configuration);
        std::string readEquation(const ptree& configuration);
        std::string readReconstruciton(const ptree& configuration);
        real readCFLNumber(const ptree& configuration);
        std::string readIntegrator(const ptree& configuration);
        boost::shared_ptr<init::InitialData> createInitialData(const ptree& configuration);
        boost::shared_ptr<io::Writer> createWriter(const ptree& configuration);
        std::string readPlatform(const ptree& configuration);
        std::string readBoundary(const ptree& configuration);

        std::string readFlux(const ptree& configuration);

        std::string basePath;
    };
} // namespace alsfvm
} // namespace config
