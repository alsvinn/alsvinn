#include "alsuq/config/Setup.hpp"
#include <cmath>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <omp.h>
#include "alsutils/log.hpp"


int main(int argc, char** argv) {
    int rank = 0;
    try {

        auto wallStart = boost::posix_time::second_clock::local_time();
        auto timeStart = boost::chrono::thread_clock::now();
        if (argc != 2) {
            std::cout << "Usage:\n\t" << argv[0] << " <inputfile.xml>" << std::endl;
            return EXIT_FAILURE;
        }

        MPI_Init(&argc, &argv);

        std::string inputfile = argv[1];

        alsuq::mpi::Config mpiConfig;

        rank = mpiConfig.getRank();
        alsutils::log::setLogFile("alsuqcli_mpi_log_" + std::to_string(mpiConfig.getRank())
                                  + ".txt");


#ifdef _OPENMPI
        ALSVINN_LOG(info) << "omp max threads= " << omp_get_max_threads() << std::endl;
#endif
        alsuq::config::Setup setup;
        auto runner = setup.makeRunner(inputfile, mpiConfig);

        ALSVINN_LOG(INFO, "Running simulator... ");


        runner->run();


        auto timeEnd = boost::chrono::thread_clock::now();
        auto wallEnd = boost::posix_time::second_clock::local_time();

        ALSVINN_LOG(INFO, "Simulation finished!" << std::endl);
        ALSVINN_LOG(INFO, "Duration: " << boost::chrono::duration_cast<boost::chrono::milliseconds>(timeEnd - timeStart).count() << " ms" << std::endl);
        ALSVINN_LOG(INFO, "Duration (wall time): " << (wallEnd - wallStart) << std::endl);

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
