#include "alsuq/config/Setup.hpp"
#include <cmath>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <omp.h>
#ifdef _WIN32
#ifndef NDEBUG
#include <float.h> // enable floating point exceptions on windows.
// see https://msdn.microsoft.com/en-us/library/aa289157(VS.71).aspx#floapoint_topic8
#endif
#endif
int main(int argc, char** argv) {
#ifdef _WIN32
#ifndef NDEBUG
    // see https://msdn.microsoft.com/en-us/library/aa289157(VS.71).aspx#floapoint_topic8
    //_clearfp();
    //unsigned int fp_control_state = _controlfp(_EM_ZERODIVIDE, _MCW_EM);
    //_controlfp(_EM_INEXACT,0);
    //feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

#endif
#endif
    try {
        auto wallStart = boost::posix_time::second_clock::local_time();
        auto timeStart = boost::chrono::thread_clock::now();
        if (argc != 2) {
            std::cout << "Usage:\n\t" << argv[0] << " <inputfile.xml>" << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "omp max threads= " << omp_get_max_threads() << std::endl;
        std::string inputfile = argv[1];

        alsuq::mpi::Config mpiConfig(argc, argv);
        alsuq::config::Setup setup;
        auto runner = setup.makeRunner(inputfile, mpiConfig);

        std::cout << "Running simulator... " << std::endl;
        std::cout << std::endl << std::endl;
        std::cout << std::numeric_limits<long double>::digits10 + 1;

        runner->run();


        auto timeEnd = boost::chrono::thread_clock::now();
        auto wallEnd = boost::posix_time::second_clock::local_time();

        std::cout << "Simulation finished!" << std::endl;
        std::cout << "Duration: " << boost::chrono::duration_cast<boost::chrono::milliseconds>(timeEnd - timeStart).count() << " ms" << std::endl;
        std::cout << "Duration (wall time): " << (wallEnd - wallStart) << std::endl;

    }
    catch (std::runtime_error& e) {
        std::cerr << "Error!" << std::endl;
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
