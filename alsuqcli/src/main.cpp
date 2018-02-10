#include "alsuq/config/Setup.hpp"
#include <cmath>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <omp.h>
#include "alsutils/log.hpp"
#include <boost/program_options.hpp>
#include "alsutils/config.hpp"
#include "alsutils/write_run_report.hpp"

int main(int argc, char** argv) {
    int rank = 0;
    try {

        auto wallStart = boost::posix_time::second_clock::local_time();
        auto timeStart = boost::chrono::thread_clock::now();
        using namespace boost::program_options;
        options_description description;
        // See http://www.boost.org/doc/libs/1_58_0/doc/html/program_options/tutorial.html
        // especially for "positional arguments"
        description.add_options()
                ("help", "Produces this help message")

        #ifdef ALSVINN_USE_MPI
                ("multi-sample", value<int>()->default_value(1),
                 "number of processors to use in the sample direction")
                ("multi-x", value<int>()->default_value(1),
                 "number of processors to use in x direction")
                ("multi-y", value<int>()->default_value(1),
                 "number of processors to use in y direction")
                ("multi-z", value<int>()->default_value(1),
                 "number of processors to use in z direction");
        #else
            ;
        #endif


        options_description hiddenDescription;
        hiddenDescription.add_options()("input", value<std::string>(), "Input xml file to use");


        options_description allOptions;
        allOptions.add(description).add(hiddenDescription);


        positional_options_description p;
        p.add("input", -1);

        variables_map vm;

        try {
            store(command_line_parser(argc, argv).options(allOptions).positional(p).run(), vm);
            notify(vm);
        } catch(std::runtime_error& error) {
            std::cout << error.what() << std::endl;
            std::cout << "Usage:\n\t" << argv[0] << " <options> <inputfile.xml>" << std::endl<< std::endl;

            std::cout << description << std::endl;

            std::exit(EXIT_FAILURE);
        }

        if (vm.count("input") == 0) {
            std::cout << "No input file given!" << std::endl;


            if (!vm.count("help")) {
                std::cout << "Usage:\n\t" << argv[0] << " <options> <inputfile.xml>" << std::endl<< std::endl;

                std::cout << description << std::endl;

                std::exit(EXIT_FAILURE);
            }
        }
        if (vm.count("help")) {
            std::cout << "Usage:\n\t" << argv[0] << " <options> <inputfile.xml>" << std::endl<< std::endl;

            std::cout << description << std::endl;

            std::exit(EXIT_FAILURE);

        }

        MPI_Init(&argc, &argv);
        alsuq::mpi::ConfigurationPtr mpiConfig(new alsuq::mpi::Configuration(MPI_COMM_WORLD));
        rank = mpiConfig->getRank();
        alsutils::log::setLogFile("alsuqcli_mpi_log_" + std::to_string(mpiConfig->getRank())
                                  + ".txt");


        alsutils::dumpInformationToLog();
        std::string inputfile = vm["input"].as<std::string>();



        const int multiX = vm["multi-x"].as<int>();
        const int multiY = vm["multi-y"].as<int>();
        const int multiZ = vm["multi-z"].as<int>();
        const int multiSample = vm["multi-sample"].as<int>();

        if (mpiConfig->getNumberOfProcesses() != multiSample * multiX * multiY * multiZ) {
            THROW("The total number of processors required is: " << multiSample*multiX*multiY*multiZ
                  << "\n"<<"The total number given was: " << mpiConfig->getNumberOfProcesses() );
        }



        ALSVINN_LOG(INFO, "omp max threads= " << omp_get_max_threads());

        if (rank == 0) {
            std::cout << "omp max threads= " << omp_get_max_threads() << std::endl;
        }

        alsuq::config::Setup setup;

        auto runner = setup.makeRunner(inputfile, mpiConfig, multiSample,
                                       alsuq::ivec3(multiX, multiY, multiZ));

        ALSVINN_LOG(INFO, "Running simulator... ");


        runner->run();


        auto timeEnd = boost::chrono::thread_clock::now();
        auto wallEnd = boost::posix_time::second_clock::local_time();

        ALSVINN_LOG(INFO, "Simulation finished!" << std::endl);
        ALSVINN_LOG(INFO, "Duration: " << boost::chrono::duration_cast<boost::chrono::milliseconds>(timeEnd - timeStart).count() << " ms" << std::endl);
        ALSVINN_LOG(INFO, "Duration (wall time): " << (wallEnd - wallStart) << std::endl);
        alsutils::writeRunReport("alsuqcli", runner->getName(),
                                 boost::chrono::duration_cast<boost::chrono::milliseconds>(timeEnd - timeStart).count(),
                                 (wallEnd - wallStart).total_milliseconds(),
                                 runner->getTimestepsPerformedTotal(), argc, argv);

    }
    catch (std::runtime_error& e) {
        ALSVINN_LOG(ERROR, "Error!");
        ALSVINN_LOG(ERROR, e.what());
        if (rank ==0 ){
            std::cerr << "Error occured in rank 0" << std::endl;
            std::cerr << e.what() << std::endl;
        }
        return EXIT_FAILURE;
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
