#include "alsfvm/config/SimulatorSetup.hpp"
#include <fstream>
#include "alsfvm/config/XMLParser.hpp"
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include "alsfvm/init/PythonInitialData.hpp"
#include "alsfvm/error/Exception.hpp"
#include "alsfvm/io/HDF5Writer.hpp"
#include "alsfvm/io/FixedIntervalWriter.hpp"
#include <boost/property_tree/xml_parser.hpp>
#include "alsfvm/init/PythonInitialData.hpp"
#include <boost/filesystem.hpp>
namespace alsfvm { namespace config {
// Example xml:
// <fvm>
//   <name>riemann</name>
//   <grid>
//     <lowerCorner>0 0 0</lowerCorner>
//     <upperCorner>1 1 0</upperCorner>
//     <dimension>10 10 1</dimension>
//   </grid>
//   <endTime>1.0</endTime>
//   <equation>euler</equation>
//   <reconstruction>none</reconstruction>
//   <cfl>auto</cfl>
//   <integrator>auto</integrator>
//   <initialData>
//     <python>riemann.py</python>
//   </initialData>
//   <writer>
//     <type>hdf5</type>
//     <basename>riemann</basename>
//     <numberOfSaves>10</numberOfSaves>
//   </writer>
// </fvm>

namespace {
    template<class T>
    vec3<T> parseVector(const std::string& vectorAsString) {
        std::vector<std::string> splitString;
        boost::split(splitString, vectorAsString, boost::is_any_of("\t "));

        T a = boost::lexical_cast<T>(splitString[0]);
        T b = boost::lexical_cast<T>(splitString[1]);
        T c = boost::lexical_cast<T>(splitString[2]);

        return vec3<T>(a, b, c);

    }
}
boost::shared_ptr<simulator::Simulator>
    SimulatorSetup::readSetupFromFile(const std::string &filename)
{
    basePath = boost::filesystem::path(filename).parent_path().string();
    std::ifstream file(filename);
    XMLParser parser;

    XMLParser::ptree configuration;
    parser.parseFile(file, configuration);

    auto grid = createGrid(configuration);
    auto boundary = readBoundary(configuration);
    real endTime = readEndTime(configuration);

    auto equation = readEquation(configuration);
    auto fluxname = readFlux(configuration);
    auto reconstruction = readReconstruciton(configuration);
    auto cfl = readCFLNumber(configuration);

    auto integrator = readIntegrator(configuration);
    auto initialData = createInitialData(configuration);
    auto writer = createWriter(configuration);

    auto platform = readPlatform(configuration);
    auto deviceConfiguration = boost::make_shared<DeviceConfiguration>(platform);

    auto memoryFactory = boost::make_shared<memory::MemoryFactory>(deviceConfiguration);
    auto volumeFactory = boost::make_shared<volume::VolumeFactory>(equation, memoryFactory);
    auto boundaryFactory = boost::make_shared<boundary::BoundaryFactory>(boundary, deviceConfiguration);
    auto numericalFluxFactory = boost::make_shared<numflux::NumericalFluxFactory>(equation, fluxname, reconstruction, deviceConfiguration);
    auto integratorFactory = boost::make_shared<integrator::IntegratorFactory>(integrator);

    auto cellComputerFactory = boost::make_shared<equation::CellComputerFactory>(platform, equation, deviceConfiguration);




    simulator::SimulatorParameters parameters;
    parameters.setCFLNumber(cfl);

    auto simulator = boost::make_shared<simulator::Simulator>(parameters,
                         grid,
                         *volumeFactory,
                         *integratorFactory,
                         *boundaryFactory,
                         *numericalFluxFactory,
                         *cellComputerFactory,
                         memoryFactory,
                         initialData,
                         endTime,
						 deviceConfiguration,
						 equation);

    simulator->addWriter(writer);

    return simulator;
}

boost::shared_ptr<grid::Grid> SimulatorSetup::createGrid(const SimulatorSetup::ptree &configuration)
{
    const ptree& gridNode =  configuration.get_child("fvm.grid");

    const std::string& lowerCornerString = gridNode.get<std::string>("lowerCorner");
    const std::string& upperCornerString = gridNode.get<std::string>("upperCorner");
    const std::string& dimensionString = gridNode.get<std::string>("dimension");


    auto lowerCorner = parseVector<real>(lowerCornerString);
    auto upperCorner = parseVector<real>(upperCornerString);
    auto dimension = parseVector<int>(dimensionString);

    return boost::make_shared<grid::Grid>(lowerCorner, upperCorner, dimension);
}

real SimulatorSetup::readEndTime(const SimulatorSetup::ptree &configuration)
{
    return configuration.get<real>("fvm.endTime");
}

std::string SimulatorSetup::readEquation(const SimulatorSetup::ptree &configuration)
{
    return configuration.get<std::string>("fvm.equation");
}

std::string SimulatorSetup::readReconstruciton(const SimulatorSetup::ptree &configuration)
{
    return configuration.get<std::string>("fvm.reconstruction");
}

real SimulatorSetup::readCFLNumber(const SimulatorSetup::ptree &configuration)
{
    auto cflString = configuration.get<std::string>("fvm.cfl");

    if (cflString == "auto") {
        auto reconstruction = readReconstruciton(configuration);
        if (reconstruction == "none") {
            return 0.9;
        } else {
            std::cout << "correct" << std::endl;
            return 0.475;
        }
    } else {
        return boost::lexical_cast<real>(cflString);
    }
}

std::string SimulatorSetup::readIntegrator(const SimulatorSetup::ptree &configuration)
{
    auto integratorString = configuration.get<std::string>("fvm.integrator");
    if (integratorString == "auto" ) {
        auto reconstruction = readReconstruciton(configuration);
        if (reconstruction == "none") {
            return "forwardeuler";
        } else {
            return "rungekutta2";
        }
    }
    return integratorString;
}

boost::shared_ptr<init::InitialData> SimulatorSetup::createInitialData(const SimulatorSetup::ptree &configuration)
{
    auto initialDataNode = configuration.get_child("fvm.initialData");

    if(initialDataNode.find("python") !=  initialDataNode.not_found()) {
        auto pythonFile = basePath + "/" + initialDataNode.get<std::string>("python");
        std::ifstream file( pythonFile);


        if (! file.good()) {
            THROW("Could not open file: " << pythonFile);
        }
        std::string pythonProgram((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());


        return boost::shared_ptr<init::InitialData>(new init::PythonInitialData(pythonProgram));
    }

    THROW("Unknown initial data.");
}

boost::shared_ptr<io::Writer> SimulatorSetup::createWriter(const SimulatorSetup::ptree &configuration)
{
    std::string type = configuration.get<std::string>("fvm.writer.type");
    std::string basename = configuration.get<std::string>("fvm.writer.basename");
    boost::shared_ptr<io::Writer> baseWriter;
    if (type == "hdf5") {
        baseWriter.reset(new io::HDF5Writer(basename));
    } else {
        THROW("Unknown writer: " << type);
    }

    const auto& writerNode = configuration.get_child("fvm.writer");
    if ( writerNode.find("numberOfSaves") != writerNode.not_found() ) {
        size_t numberOfSaves = writerNode.get<size_t>("numberOfSaves");
        real endTime = readEndTime(configuration);
        real timeInterval = endTime / numberOfSaves;
        return boost::shared_ptr<io::Writer>(new io::FixedIntervalWriter(baseWriter, timeInterval, endTime));
    }
    return baseWriter;
}

std::string SimulatorSetup::readPlatform(const SimulatorSetup::ptree &configuration)
{
    return configuration.get<std::string>("fvm.platform");
}

std::string SimulatorSetup::readBoundary(const SimulatorSetup::ptree &configuration)
{
    return configuration.get<std::string>("fvm.boundary");
}

std::string SimulatorSetup::readFlux(const SimulatorSetup::ptree &configuration)
{
    return configuration.get<std::string>("fvm.flux");
}

}
}
