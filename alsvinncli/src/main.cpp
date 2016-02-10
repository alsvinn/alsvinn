#include <alsfvm/config/SimulatorSetup.hpp>
#include <cmath>
#include <boost/chrono.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <omp.h>
int main(int argc, char** argv) {
	try {
	        auto wallStart = boost::posix_time::second_clock::local_time();
		auto timeStart = boost::chrono::thread_clock::now();
		if (argc != 2) {
			std::cout << "Usage:\n\t" << argv[0] << " <inputfile.xml>" << std::endl;
			return EXIT_FAILURE;
		}
		std::cout << "omp max threads= " << omp_get_max_threads() << std::endl;
		std::string inputfile = argv[1];

		alsfvm::config::SimulatorSetup setup;

		auto simulator = setup.readSetupFromFile(inputfile);

		std::cout << "Running simulator... " << std::endl;
		std::cout << std::endl << std::endl;
		std::cout << std::numeric_limits<long double>::digits10 + 1;

		simulator->callWriters();
		


		int lastPercentSeen = -1;
        size_t timestepsPerformed = 0;
		while (!simulator->atEnd()) {
			simulator->performStep();
            timestepsPerformed++;
			int percentDone = std::round(100.0 * simulator->getCurrentTime() / simulator->getEndTime());
			if (percentDone != lastPercentSeen) {
				std::cout << "\rPercent done: " << percentDone << std::flush;
				lastPercentSeen = percentDone;
			}

		}
		std::cout << std::endl << std::endl;
        std::cout << "timesteps = " << timestepsPerformed << std::endl;
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
