#include <alsfvm/config/SimulatorSetup.hpp>
#include <cmath>
#ifdef ALSVINN_USE_MPI
#include <mpi.h>
#include "alsutils/mpi/safe_call.hpp"
#endif

#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/program_options.hpp>
#include <omp.h>
#include "alsutils/log.hpp"
#include "alsutils/config.hpp"
#ifdef _WIN32 
#ifndef NDEBUG
#include <float.h> // enable floating point exceptions on windows.
// see https://msdn.microsoft.com/en-us/library/aa289157(VS.71).aspx#floapoint_topic8
#endif
#endif
#include <cstdlib>
#ifdef ALSVINN_HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif


int main(int argc, char** argv) {
  setenv("MPICH_RDMA_ENABLED_CUDA", "1", 1);
  setenv("MV2_USE_CUDA", "1", 1);


	try {

        using namespace boost::program_options;
        options_description description;
        // See http://www.boost.org/doc/libs/1_58_0/doc/html/program_options/tutorial.html
        // especially for "positional arguments"
        description.add_options()
                ("help", "Produces this help message")

        #ifdef ALSVINN_USE_MPI
                ("automatic-x,x", "Divides all cores available in the X direction")
                ("automatic-y,y", "Divides all cores available in the Y direction")
                ("automatic-z,z", "Divides all cores available in the Z direction")
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

#ifdef ALSVINN_USE_MPI
      int mpiRank;

  MPI_SAFE_CALL(MPI_Init(NULL, NULL));


      MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank));

      int numberOfProcessors;

      MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &numberOfProcessors));


        alsutils::log::setLogFile("alsvinncli_mpi_log_" + std::to_string(mpiRank)
                                  + ".txt");
        ALSVINN_LOG(INFO, "MPI enabled");
#else
        ALSVINN_LOG(INFO, "MPI disabled");

        int mpiRank = 0;
#endif
        auto wallStart = boost::posix_time::second_clock::local_time();

		auto timeStart = boost::chrono::thread_clock::now();

#ifdef _OPENMP
        ALSVINN_LOG(INFO, "omp max threads= " << omp_get_max_threads());
#endif
        std::string inputfile = vm["input"].as<std::string>();


		alsfvm::config::SimulatorSetup setup;
    ALSVINN_LOG(INFO, "Git commit " << alsutils::getVersionControlID());
#ifdef ALSVINN_USE_MPI

        int multiX = vm["multi-x"].as<int>();
        int multiY = vm["multi-y"].as<int>();
        int multiZ = vm["multi-z"].as<int>();


        if (vm.count("automatic-x")) {
           multiX = numberOfProcessors;
           multiY = 1;
           multiZ = 1;

        }

        if (vm.count("automatic-y")) {
            multiX = 1;
            multiY = numberOfProcessors;
            multiZ = 1;

        }

        if (vm.count("automatic-z")) {
            multiX = 1;
            multiY = 1;
            multiZ = numberOfProcessors;

        }
        if (numberOfProcessors != multiX*multiY*multiZ) {
            THROW("The total number of processors required is: " << multiX*multiY*multiZ
                  << "\n"<<"The total number given was: " << numberOfProcessors);
        }
        setup.enableMPI(MPI_COMM_WORLD, multiX, multiY, multiZ);
#endif

        auto simulatorPair = setup.readSetupFromFile(inputfile);

        auto simulator = simulatorPair.first;
        simulator->setInitialValue(simulatorPair.second);

        if (mpiRank == 0) {
            std::cout << "Running simulator... " << std::endl;
            std::cout << std::endl;
            std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
        }

		simulator->callWriters();
		


		int lastPercentSeen = -1;
        size_t timestepsPerformed = 0;
        
		while (!simulator->atEnd()) {

			simulator->performStep();
            timestepsPerformed++;
			int percentDone = std::round(100.0 * simulator->getCurrentTime() / simulator->getEndTime());
			if (percentDone != lastPercentSeen) {
                if (mpiRank == 0) {
                    std::cout << "#" << std::flush;
                }
				lastPercentSeen = percentDone;
			}

		}
        simulator->finalize();
        if (mpiRank == 0) {
            std::cout << std::endl << std::endl;
        }

        ALSVINN_LOG(INFO, "timesteps = " << timestepsPerformed);
		auto timeEnd = boost::chrono::thread_clock::now();
		auto wallEnd = boost::posix_time::second_clock::local_time();

        ALSVINN_LOG(INFO, "Simulation finished!")
        ALSVINN_LOG(INFO, "Duration: " << boost::chrono::duration_cast<boost::chrono::milliseconds>(timeEnd - timeStart).count() << " ms");
        ALSVINN_LOG(INFO, "Duration (wall time): " << (wallEnd - wallStart));


	}
	catch (std::runtime_error& e) {
        ALSVINN_LOG(ERROR, "Error!" << std::endl
                    << e.what() << std::endl);

        std::cerr << "An error occured." << std::endl;
        std::cerr << "The error message was:"<< std::endl;
        std::cerr << e.what() << std::endl;


		return EXIT_FAILURE;
	}

     MPI_SAFE_CALL(MPI_Finalize());
    return EXIT_SUCCESS;
}
