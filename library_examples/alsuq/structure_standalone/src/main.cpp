/** [headers]*/
// STL headers (not needed from Alsvinn, just this example)
#include <iostream>
#include <random>

// Alsvinn headers
#include <alsfvm/io/WriterFactory.hpp>
#include <alsuq/stats/StatisticsFactory.hpp>
#include <alsfvm/volume/make_volume.hpp>
#include <alsfvm/volume/VolumePair.hpp>
/** [headers]*/

/** [simulator_info]*/
// It is probably a good idea to add version information to the file
#define SIMULATOR_NAME "3druns"
#define SIMULATOR_VERSION "0.0.1"
#define RAW_DATA_GENERATED_BY "alsvinn"
/** [simulator_info]*/

// file reading
#include <alsfvm/io/netcdf_utils.hpp>
#include <netcdf.h>

// program options
#include <boost/program_options.hpp>

// logging
#include "alsutils/log.hpp"

#include <chrono>
#include "alsutils/mpi/set_cuda_device.hpp"
#include "git_config.hpp"



boost::property_tree::ptree getAttributes(const std::string& filename) {

    boost::property_tree::ptree attributes;

    alsfvm::io::netcdf_raw_ptr file;
    NETCDF_SAFE_CALL(nc_open(filename.c_str(), NC_NOWRITE, &file));

    int numberOfAttributes = 0;

    NETCDF_SAFE_CALL(nc_inq_natts(file, &numberOfAttributes));

    for (int attributeNumber = 0; attributeNumber < numberOfAttributes;
        ++attributeNumber) {
        std::vector<char> attributeNameVector(NC_MAX_NAME);
        NETCDF_SAFE_CALL(nc_inq_attname(file, NC_GLOBAL, attributeNumber,
                attributeNameVector.data()));

        std::string attributeName(attributeNameVector.begin(),
            attributeNameVector.end());
        nc_type attributeType;
        NETCDF_SAFE_CALL(nc_inq_atttype(file, NC_GLOBAL, attributeName.c_str(),
                &attributeType));

        if (attributeType != NC_STRING && attributeType != NC_CHAR) {
            continue;
        }

        size_t attributeLength = 0;

        NETCDF_SAFE_CALL(nc_inq_attlen(file, NC_GLOBAL, attributeName.c_str(),
                &attributeLength));


        std::vector<char> attributeAsVector(attributeLength + 1);
        NETCDF_SAFE_CALL(nc_get_att_text(file, NC_GLOBAL, attributeName.c_str(),
                attributeAsVector.data()));



        std::string attributeValue(attributeAsVector.begin(),
            attributeAsVector.end());
        attributes.put(attributeName, attributeValue);
    }

    NETCDF_SAFE_CALL(nc_close(file));
    return attributes;
}

void addWriters(alsuq::stats::StatisticsFactory::StatisticsPointer
    statisticsPointer,
    const std::string& outputName,
    const std::string& writerType,
    boost::property_tree::ptree extraAttributes,
    const std::string& allCommandLineArguments) {
    alsfvm::io::WriterFactory factory;
    boost::property_tree::ptree tree;
    alsfvm::io::Parameters parameters(tree);

    // this is a general routine, some statistics have more than one output
    // (eg. meanvar saves mean and var), we need to supply a writer for each output
    for (auto statsName :
        statisticsPointer->getStatisticsNames()) { // loop through each output
        auto writer = factory.createWriter(writerType, outputName + "_" + statsName,
                parameters);
        // It is a good idea to add some attributes to the writer to
        // know mark the data
        boost::property_tree::ptree attributes;
        attributes.put("generatedBy", "alsvinn/library_examples/strcture_standalone");
        attributes.put("commandLineArguments", allCommandLineArguments);
        attributes.put("structureGitCommit", getGitCommit());
        attributes.put("structureGitVersionStatus", getGitVersionStatus());


        writer->addAttributes("standAloneProgramInformation", attributes);

        writer->addAttributes("fromFile", extraAttributes);
        // add writer to statistics
        statisticsPointer->addWriter(statsName, writer);
    }
}


/** [makeStructureFunction]*/
alsuq::stats::StatisticsFactory::StatisticsPointer makeStructureFunction(
    double p, int numberOfH, int numberOfSamples, const std::string& platform,
    const alsuq::mpi::ConfigurationPtr mpiConfiguration) {
    /** [makeStructureFunction]*/
    /** [factoryInstance]*/
    alsuq::stats::StatisticsFactory statisticsFactory;
    /** [factoryInstance]*/

    /** [parameters]*/
    boost::property_tree::ptree properties;
    properties.put("p", p);
    properties.put("numberOfH", numberOfH);
    alsuq::stats::StatisticsParameters parameters(properties);
    parameters.setNumberOfSamples(numberOfSamples);
    parameters.setMpiConfiguration(mpiConfiguration);
    /** [parameters]*/

    /** [createStatistics]*/
    auto statistics = statisticsFactory.makeStatistics(platform, "structure_cube",
            parameters);

    return statistics;
    /** [createStatistics]*/
}


alsfvm::simulator::TimestepInformation getTimestepInformation(
    const std::string& filename) {

    double time = 0;
    alsfvm::io::netcdf_raw_ptr file;
    NETCDF_SAFE_CALL(nc_open(filename.c_str(), NC_NOWRITE, &file));

    int variableId = 0;
    int variableStatus = nc_inq_varid(file, "time", &variableId);

    if (variableStatus == NC_NOERR) {
        int variableLength = 0;

        nc_type variableType;
        NETCDF_SAFE_CALL(nc_inq_vartype(file, variableId, &variableType));

        if (variableType == NC_DOUBLE) {
            NETCDF_SAFE_CALL(nc_get_var_double(file, variableId, &time));

            ALSVINN_LOG(INFO, "Time is " << time)
        }
    }

    return alsfvm::simulator::TimestepInformation(time, 0);
}

alsfvm::volume::VolumePair getSample(const std::string& platform,
    const std::string& equation,
    int sample,
    const std::string& filename,
    int nx, int ny, int nz) {
    auto start = std::chrono::high_resolution_clock::now();
    using namespace alsfvm::io;
    netcdf_raw_ptr file;

    NETCDF_SAFE_CALL(nc_open(filename.c_str(), NC_NOWRITE, &file));

    auto conservedVolume = alsfvm::volume::makeConservedVolume(platform, equation, {nx, ny, nz},
            0);

    for (int var = 0; var < conservedVolume->getNumberOfVariables(); ++var) {
        auto name = conservedVolume->getName(var);
        auto variableName = std::string("sample_") + std::to_string(
                sample) + "_" + name;


        netcdf_raw_ptr varid;
        NETCDF_SAFE_CALL(nc_inq_varid(file, variableName.c_str(), &varid));

        netcdf_raw_ptr netcdftype;
        NETCDF_SAFE_CALL(nc_inq_typeid(file, variableName.c_str(), &netcdftype));

        std::vector<::alsfvm::real> buffer (nx * ny * nz);

        if (netcdftype == NC_DOUBLE) {
            std::vector<double> bufferTmp(nx * ny * nz);

            NETCDF_SAFE_CALL(nc_get_var_double(file, varid, bufferTmp.data()));

            std::copy(bufferTmp.begin(), bufferTmp.end(), buffer.begin());

        } else if (netcdftype == NC_FLOAT) {
            std::vector<float> bufferTmp(nx * ny * nz);

            NETCDF_SAFE_CALL(nc_get_var_float(file, varid, bufferTmp.data()));

            std::copy(bufferTmp.begin(), bufferTmp.end(), buffer.begin());

        }

        conservedVolume->getScalarMemoryArea(var)->copyFromHost(buffer.data(),
            buffer.size());
    }

    NETCDF_SAFE_CALL(nc_close(file));

    ALSVINN_LOG(INFO, "Read sample")

    auto end = std::chrono::high_resolution_clock::now();

    ALSVINN_LOG(INFO, "Reading sample took: " <<
        std::chrono::duration_cast<std::chrono::duration<double>>
        (end - start).count() << " s")

    return alsfvm::volume::VolumePair(conservedVolume);

}

int main(int argc, char** argv) {
    alsutils::mpi::setCudaDevice();
    MPI_Init(&argc, &argv);




    using namespace boost::program_options;
    options_description description;
    std::string filenameInput;
    std::string filenameOutput;

    std::string equation;
    int nx, ny, nz;
    int numberOfH;
    int p;
    int numberOfSamples;
    std::string platform;
    std::string bc;

    // See http://www.boost.org/doc/libs/1_58_0/doc/html/program_options/tutorial.html
    // especially for "positional arguments"
    description.add_options()
    ("help", "Produces this help message")

    ("input-filename,i", value<std::string>(&filenameInput)->required(),
        "Input filename (only supports NetCDF)")
    ("output-filename,o", value<std::string>(&filenameOutput)->required(),
        "Output filename (only supports NetCDF)")
    ("samples", value<int>(&numberOfSamples)->default_value(1), "Number of samples")
    ("equation", value<std::string>(&equation)->default_value("euler3"),
        "Equation to use")
    ("p", value<int>(&p)->default_value(1), "Value of p")
    ("number-of-h", value<int>(&numberOfH)->default_value(1),
        "Number of h values to use")
    ("nx", value<int>(&nx)->required(), "Resolution x direction")
    ("ny", value<int>(&ny)->required(), "Resolution y direction")
    ("nz", value<int>(&nz)->required(), "Resolution z direction")
    ("boundary-condition", value<std::string>(&bc)->default_value("periodic"),
        "Boundary condition")
    ("platform", value<std::string>(&platform)->default_value("cpu"),
        "Platform to use (either cpu or cuda)");

    variables_map vm;

    try {
        store(command_line_parser(argc, argv).options(description).run(),
            vm);
        notify(vm);
    } catch (std::runtime_error& error) {
        std::cout << error.what() << std::endl;
        std::cout << "Usage:\n\t" << argv[0] << " <options>" <<
            std::endl << std::endl;

        std::cout << description << std::endl;

        std::exit(EXIT_FAILURE);

    } catch (...) {
        std::cout << "Usage:\n\t" << argv[0] << " <options>" <<
            std::endl << std::endl;

        std::cout << description << std::endl;

        std::exit(EXIT_FAILURE);
    }

    std::stringstream allCommandLineArguments;

    for (int arg = 0; arg < argc; ++arg) {
        allCommandLineArguments << argv[arg] << " ";
    }


    const alsfvm::rvec3 lower = {0, 0, 0};
    const alsfvm::rvec3 upper = {1, static_cast<alsfvm::real>(ny > 1), static_cast<alsfvm::real>(nz > 1)};



    const alsfvm::simulator::TimestepInformation timestepInformation =
        getTimestepInformation(filenameInput);

    // Output
    const std::string writerType = "netcdf";

    // This is for the MPI setup
    auto mpiConfiguration = std::make_shared<alsuq::mpi::Configuration>
        (MPI_COMM_WORLD, platform);

    //ALSVINN_LOG(INFO, "Reading from " << filenameInput);
    auto rank = mpiConfiguration->getRank();
    auto numberOfProcessors = mpiConfiguration->getNumberOfProcesses();

    if (numberOfSamples % numberOfProcessors != 0) {
        std::cerr << "Number of processors must be a factor in number of samples" <<
            std::endl;
        std::cerr << "Given\n"
            << "\tnumberOfSamples = " << numberOfSamples << "\n"
            << "\tNumberOfProcessors = " << numberOfProcessors << "\n"
            << std::endl;
    }

    if (numberOfSamples < numberOfProcessors) {
        std::cerr << "Number of processors larger than number of samples" << std::endl;
        std::cerr << "Given\n"
            << "\tnumberOfSamples = " << numberOfSamples << "\n"
            << "\tNumberOfProcessors = " << numberOfProcessors << "\n"
            << std::endl;
    }

    auto samplesPerProcessor = numberOfSamples / numberOfProcessors;



    auto attributes = getAttributes(filenameInput);
    auto statistics = makeStructureFunction(p, numberOfH, numberOfSamples, platform,
            mpiConfiguration);
    addWriters(statistics, filenameOutput, writerType, attributes,
        allCommandLineArguments.str());

    alsfvm::grid::Grid grid(lower, upper, {nx, ny, nz});


    const auto sampleStart = rank * samplesPerProcessor;
    const auto sampleEnd = (rank + 1) * samplesPerProcessor;


    alsutils::log::setLogFile("structure_standalone_mpi_log_"
        + std::to_string(p) + "_"
        + std::to_string(timestepInformation.getCurrentTime()) + "_"
        + std::to_string(rank)
        + ".txt");

    for (int sample = sampleStart; sample < sampleEnd; ++sample) {
        auto start = std::chrono::high_resolution_clock::now();
        ALSVINN_LOG(INFO, "sample: " << sample)
        auto volumes = getSample(platform, equation, sample, filenameInput, nx, ny, nz);

        statistics->write(*volumes.getConservedVolume(),
            grid,
            timestepInformation);
        auto end = std::chrono::high_resolution_clock::now();

        ALSVINN_LOG(INFO, "Computing sample took: " <<
            std::chrono::duration_cast<std::chrono::duration<double>>
            (end - start).count() << " s")
    }

    statistics->combineStatistics();

    if (rank == 0) {
        statistics->finalizeStatistics();

        statistics->writeStatistics(grid);
    }

    MPI_Finalize();

}
