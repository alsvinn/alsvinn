#include "alsfvm/types.hpp"
#include "alsfvm/config/SimulatorSetup.hpp"
#include <fstream>
#include "alsfvm/config/XMLParser.hpp"
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include "alsfvm/init/PythonInitialData.hpp"
#include "alsutils/error/Exception.hpp"
#include "alsfvm/io/HDF5Writer.hpp"
#include "alsfvm/io/FixedIntervalWriter.hpp"
#include "alsfvm/io/CoarseGrainingIntervalWriter.hpp"
#include <boost/property_tree/xml_parser.hpp>
#include "alsfvm/init/PythonInitialData.hpp"
#include <boost/filesystem.hpp>
#include "alsfvm/equation/euler/EulerParameters.hpp"
#include "alsfvm/diffusion/DiffusionFactory.hpp"
#include <set>
#include "alsutils/log.hpp"

#ifdef ALSVINN_USE_MPI
#include "alsfvm/mpi/domain/CartesianDecomposition.hpp"
#include "alsfvm/io/MpiWriterFactory.hpp"
#endif

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

std::pair<alsfvm::shared_ptr<simulator::Simulator>,
alsfvm::shared_ptr<init::InitialData> >
SimulatorSetup::readSetupFromFile(const std::string &filename)
{
    if (!boost::filesystem::exists(filename)) {
        THROW("Input file does not exist\n" << filename );
    }
    basePath = boost::filesystem::path(boost::filesystem::absolute(filename)).parent_path().string();
    std::ifstream file(filename);
    XMLParser parser;


    XMLParser::ptree configurationBase;
    parser.parseFile(file, configurationBase);
    auto configuration = configurationBase.get_child("config");


    std::set<std::string> supportedNodes =
    {
        "name", "platform", "boundary", "flux", "endTime", "equation", "equationParameters",
        "reconstruction", "cfl", "integrator", "initialData", "writer", "grid", "diffusion"
    };

    for (auto node : configuration.get_child("fvm")) {
        if (supportedNodes.find(node.first) == supportedNodes.end()) {
            THROW("Unsupported configuration option " << node.first);
        }
    }
    auto grid = createGrid(configuration); 
    auto boundary = readBoundary(configuration);

#ifdef ALSVINN_USE_MPI
    mpi::CellExchangerPtr cellExchangerPtr;
    if (useMPI) {
        auto domainInformation = decomposeGrid(grid);
        grid = domainInformation->getGrid();
        cellExchangerPtr = domainInformation->getCellExchanger();
    }
#endif

    real endTime = readEndTime(configuration);

    auto equation = readEquation(configuration);
    auto fluxname = readFlux(configuration);
    auto reconstruction = readReconstruciton(configuration);
    auto cfl = readCFLNumber(configuration);

    auto integrator = readIntegrator(configuration);
    auto initialData = createInitialData(configuration);
    auto writer = createWriter(configuration);

    auto platform = readPlatform(configuration);
    auto deviceConfiguration = alsfvm::make_shared<DeviceConfiguration>(platform);


    alsfvm::shared_ptr<simulator::SimulatorParameters>
            parameters(new simulator::SimulatorParameters(equation, platform));
    readEquationParameters(configuration, *parameters);
    parameters->setCFLNumber(cfl);

    auto memoryFactory = alsfvm::make_shared<memory::MemoryFactory>(deviceConfiguration);
    auto volumeFactory = alsfvm::make_shared<volume::VolumeFactory>(equation, memoryFactory);
    auto boundaryFactory = alsfvm::make_shared<boundary::BoundaryFactory>(boundary, deviceConfiguration);
    auto numericalFluxFactory = alsfvm::make_shared<numflux::NumericalFluxFactory>(equation, fluxname, reconstruction, parameters, deviceConfiguration);
    auto integratorFactory = alsfvm::make_shared<integrator::IntegratorFactory>(integrator);

    auto cellComputerFactory = alsfvm::make_shared<equation::CellComputerFactory>(parameters, deviceConfiguration);


    auto diffusionOperator = createDiffusion(configuration, *grid, *parameters, deviceConfiguration, memoryFactory, *volumeFactory);



    auto simulator = alsfvm::make_shared<simulator::Simulator>(*parameters,
                                                               grid,
                                                               *volumeFactory,
                                                               *integratorFactory,
                                                               *boundaryFactory,
                                                               *numericalFluxFactory,
                                                               *cellComputerFactory,
                                                               memoryFactory,
                                                               endTime,
                                                               deviceConfiguration,
                                                               equation,
                                                               diffusionOperator);

    simulator->setCellExchanger(cellExchangerPtr);

    if (writer) {
        simulator->addWriter(writer);
    }

    auto timestepAdjuster = alsfvm::dynamic_pointer_cast<integrator::TimestepAdjuster>(writer);
    if (timestepAdjuster) {
        simulator->addTimestepAdjuster(timestepAdjuster);
    }

    return std::make_pair(simulator, initialData);
}

void SimulatorSetup::setWriterFactory(std::shared_ptr<io::WriterFactory> writerFactory)
{
    this->writerFactory = writerFactory;
}

#ifdef ALSVINN_USE_MPI
void SimulatorSetup::enableMPI(MPI_Comm communicator, int multiX, int multiY, int multiZ)
{
    useMPI = true;
    mpiConfiguration = alsfvm::make_shared<mpi::Configuration>(communicator);
    this->multiX = multiX;
    this->multiY = multiY;
    this->multiZ = multiZ;

    alsfvm::shared_ptr<io::WriterFactory> writerFactory;
    writerFactory.reset(new io::MpiWriterFactory(mpiConfiguration));
    setWriterFactory(writerFactory);
}
#endif


alsfvm::shared_ptr<grid::Grid> SimulatorSetup::createGrid(const SimulatorSetup::ptree &configuration)
{
    const ptree& gridNode =  configuration.get_child("fvm.grid");

    const std::string& lowerCornerString = gridNode.get<std::string>("lowerCorner");
    const std::string& upperCornerString = gridNode.get<std::string>("upperCorner");
    const std::string& dimensionString = gridNode.get<std::string>("dimension");


    auto lowerCorner = parseVector<real>(lowerCornerString);
    auto upperCorner = parseVector<real>(upperCornerString);
    auto dimension = parseVector<int>(dimensionString);

    auto boundaryName = readBoundary(configuration);

    std::array<boundary::Type, 6> boundaryConditions;

    if (boundaryName == "periodic") {
        boundaryConditions = boundary::allPeriodic();
    } else if (boundaryName == "neumann") {
        boundaryConditions = boundary::allNeumann();
    } else {
        THROW("Unknown boundary conditions " << boundaryName);
    }

    return alsfvm::make_shared<grid::Grid>(lowerCorner, upperCorner, dimension,
                                           boundaryConditions);
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
    auto reconstruction = configuration.get<std::string>("fvm.reconstruction");
    boost::trim(reconstruction);
    return reconstruction;
}

real SimulatorSetup::readCFLNumber(const SimulatorSetup::ptree &configuration)
{
    auto cflString = configuration.get<std::string>("fvm.cfl");

    if (cflString == "auto") {
        auto reconstruction = readReconstruciton(configuration);
        if (reconstruction == "none") {
            return 0.9;
        } else {
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

alsfvm::shared_ptr<init::InitialData> SimulatorSetup::createInitialData(const SimulatorSetup::ptree &configuration)
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

        auto parameters = readParameters(configuration);
        return alsfvm::shared_ptr<init::InitialData>(new init::PythonInitialData(pythonProgram, parameters));
    }

    THROW("Unknown initial data.");
}

alsfvm::shared_ptr<io::Writer> SimulatorSetup::createWriter(const SimulatorSetup::ptree &configuration)
{
    auto fvmNode  = configuration.get_child("fvm");
    if (fvmNode.find("writer") != fvmNode.not_found()) {

        std::string type = configuration.get<std::string>("fvm.writer.type");
        std::string basename = configuration.get<std::string>("fvm.writer.basename");
        auto baseWriter = writerFactory->createWriter(type, basename);
        ALSVINN_LOG(INFO, "Adding writer " << basename);
        const auto& writerNode = configuration.get_child("fvm.writer");
        if ( writerNode.find("numberOfSaves") != writerNode.not_found() ) {
            size_t numberOfSaves = writerNode.get<size_t>("numberOfSaves");
            real endTime = readEndTime(configuration);
            real timeInterval = endTime / numberOfSaves;

            if (writerNode.find("numberOfCoarseSaves") != writerNode.not_found()) {
                int numberOfCoarseSaves = writerNode.get<size_t>("numberOfCoarseSaves");
                int numberOfSkips = writerNode.get<size_t>("numberOfSkips");
                return alsfvm::shared_ptr<io::Writer>(new io::CoarseGrainingIntervalWriter(baseWriter,
                                                                                           timeInterval,
                                                                                           numberOfCoarseSaves,
                                                                                           endTime,
                                                                                           numberOfSkips));
            }

            return alsfvm::shared_ptr<io::Writer>(new io::FixedIntervalWriter(baseWriter, timeInterval, endTime));
        }

        return baseWriter;
    }
    return alsfvm::shared_ptr<io::Writer>();
}

std::string SimulatorSetup::readPlatform(const SimulatorSetup::ptree &configuration)
{
    return configuration.get<std::string>("fvm.platform");
}

std::string SimulatorSetup::readBoundary(const SimulatorSetup::ptree &configuration)
{
    return configuration.get<std::string>("fvm.boundary");
}

init::Parameters SimulatorSetup::readParameters(const SimulatorSetup::ptree& configuration)
{
    init::Parameters parameters;
    auto fvmNode = configuration.get_child("fvm");
    // first read in all equation parameters:
    if (fvmNode.find("equationParameters") != fvmNode.not_found()) {
        auto equationParameters = fvmNode.get_child("equationParameters");
        for (auto equationParameter : equationParameters) {
            parameters.addParameter(equationParameter.first, { real(std::atof(equationParameter.second.data().c_str())) });
        }
    }

    auto initial = fvmNode.get_child("initialData");
    if (initial.find("parameters") != initial.not_found()) {
        auto initParameters = initial.get_child("parameters");

        for (auto parameter : initParameters) {
            auto name = parameter.second.get<std::string>("name");
            auto length = parameter.second.get<int>("length");
            std::vector<real    > value;

            if (length == 1) {
                value.resize(1);
                value[0] = parameter.second.get<real>("value");
            }
            else {
                ALSVINN_LOG(INFO, "Reading parameter array");
                // We are given an array, and we read everything in
                for (auto valueItem : parameter.second.get_child("values")) {

                    if (boost::iequals(valueItem.first, "value")) {
                        value.push_back(valueItem.second.get_value<real>());
                        ALSVINN_LOG(INFO, "Read value " << value.back());
                    }
                }

                if (value.size() == 1) {
                    // We only got a single value
                    value.resize(length, value[0]);
                }
            }

            parameters.addParameter(name, value);
        }
    }
    return parameters;
}

void SimulatorSetup::readEquationParameters(const SimulatorSetup::ptree &configuration, simulator::SimulatorParameters &parameters)
{

    auto& equationParameters = parameters.getEquationParameters();

    auto fvmNode = configuration.get_child("fvm");
    if (fvmNode.find("equationParameters") != fvmNode.not_found()) {
        const auto equationName = readEquation(configuration);
        if (equationName== "euler1" || equationName == "euler2" || equationName == "euler3") {

            auto& eulerParameters = static_cast<equation::euler::EulerParameters&>(equationParameters);

            real gamma = configuration.get<real>("fvm.equationParameters.gamma");
            eulerParameters.setGamma(gamma);
        } else if (equationName == "burgers") {
            // we do not need to do anything.

        }
    }
}

alsfvm::shared_ptr<diffusion::DiffusionOperator> SimulatorSetup::createDiffusion(const SimulatorSetup::ptree& configuration,
                                                                                 const grid::Grid& grid,
                                                                                 const simulator::SimulatorParameters& simulatorParameters,
                                                                                 alsfvm::shared_ptr<DeviceConfiguration> deviceConfiguration,
                                                                                 alsfvm::shared_ptr<memory::MemoryFactory>& memoryFactory,
                                                                                 volume::VolumeFactory& volumeFactory)
{
    // Should look like this:
    //   <diffusion>
    //     <name>name</name>
    //     <reconstruction>reconstruction</reconstruction>
    //  </diffusion>
    auto fvmNode = configuration.get_child("fvm");
    std::string name = "none";
    std::string reconstruction = "none";
    if (fvmNode.find("diffusion") != fvmNode.not_found()) {

        name = configuration.get<std::string>("fvm.diffusion.name");
        boost::trim(name);
        reconstruction = configuration.get<std::string>("fvm.diffusion.reconstruction");
        boost::trim(reconstruction);
    }

    diffusion::DiffusionFactory diffusionFactory;

    return diffusionFactory.createDiffusionOperator(readEquation(configuration), name, reconstruction, grid, simulatorParameters,
                                                    deviceConfiguration, memoryFactory, volumeFactory);
}

std::string SimulatorSetup::readFlux(const SimulatorSetup::ptree &configuration)
{
    return configuration.get<std::string>("fvm.flux");
}

mpi::domain::DomainInformationPtr SimulatorSetup::decomposeGrid(const alsfvm::shared_ptr<grid::Grid> &grid)
{
    // for now we assume we have a cartesian grid
    mpi::domain::CartesianDecomposition cartesianDecomposition(multiX, multiY, multiZ);

    return cartesianDecomposition.decompose(mpiConfiguration, *grid);

}

}
                 }
