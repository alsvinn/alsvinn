#include <iostream>
#include <alsfvm/io/WriterFactory.hpp>
#include <alsuq/stats/StatisticsFactory.hpp>
#include <alsfvm/volume/make_volume.hpp>
#include <alsfvm/volume/VolumePair.hpp>
#include <random>

void addWriters(alsuq::stats::StatisticsFactory::StatisticsPointer statisticsPointer,
		const std::string& outputName,
		const std::string& writerType) {
  alsfvm::io::WriterFactory factory;
  for (auto statsName :  statisticsPointer->getStatisticsNames()) {
    auto writer = factory.createWriter(writerType, outputName + "_" + statsName);
    statisticsPointer->addWriter(statsName, writer);
  }
}
  
		

alsuq::stats::StatisticsFactory::StatisticsPointer
makeStructureFunction(double p, int numberOfH, int numberOfSamples, const std::string& platform,
		      const alsuq::mpi::ConfigurationPtr mpiConfiguration) {
  alsuq::stats::StatisticsFactory statisticsFactory;

  boost::property_tree::ptree properties;
  properties.put("p", 2);
  properties.put("numberOfH", 32);
  alsuq::stats::StatisticsParameters parameters(properties);
  parameters.setNumberOfSamples(numberOfSamples);
  parameters.setMpiConfiguration(mpiConfiguration);
  
  auto statistics = statisticsFactory.makeStatistics(platform, "structure_cube", parameters);

  return statistics;
}

alsfvm::volume::VolumePair getSample(const std::string& platform,
				     const std::string& equation,
				     int sample, int nx, int ny, int nz) {

  
  auto conservedVolume = alsfvm::volume::makeConservedVolume(platform, equation, {nx, ny, nz},
							     0);

  auto extraVolume = alsfvm::volume::makeExtraVolume(platform, equation, {nx, ny, nz}, 0);


  // Fill conservedVolume with some junk
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0, 1);
  for (int z = 0; z < nz; ++z) {
    for (int y = 0; y < ny; ++y) {
      for (int x = 0; x < nx; ++x) {
	(*conservedVolume)["u"](x, y, z) = distribution(generator);
      }
    }
  }
  
  return alsfvm::volume::VolumePair(conservedVolume, extraVolume);
}

int main(int argc, char** argv) {
  
  MPI_Init(&argc, &argv);


  // Number of samples to use
  const int numberOfSamples = 32;

  // Parameters to the structure function,
  // p is the exponent to use,
  // numberOfH is the number of grid cells to use
  const double p = 2;
  const int numberOfH = 32;

  // Grid setup
  const int nx = 64;
  const int ny = 64;
  const int nz = 1;
  const alsfvm::rvec3 lower = {0,0,0};
  const alsfvm::rvec3 upper = {1,1,0};

  // This information need not be correct, but it can make the output files nicer
  // (with more information stored)
  const double currentTime = 1.2;
  const int numberOfTimestepsPerformed = 42;
  const alsfvm::simulator::TimestepInformation timestepInformation(currentTime, numberOfTimestepsPerformed);

  // Output
  const std::string writerType = "netcdf";
  const std::string outputName = "random_noise";

  // Just make burgers for now
  const std::string equation = "burgers";

  // We stick to cpu
  const std::string platform = "cpu";


  // This is for the MPI setup
  auto mpiConfiguration = std::make_shared<alsuq::mpi::Configuration>(MPI_COMM_WORLD, platform);

  

  auto statistics = makeStructureFunction(p, numberOfH, numberOfSamples, platform, mpiConfiguration);
  addWriters(statistics, outputName, writerType);

  alsfvm::grid::Grid grid(lower, upper, {nx, ny, nz});



  for (int sample = 0; sample < numberOfSamples; ++sample) {
    auto volumes = getSample(platform, equation, sample, nx, ny, nz);
    statistics->write(*volumes.getConservedVolume(),
		      *volumes.getExtraVolume(),
		      grid,
		      timestepInformation);
  }

  statistics->combineStatistics();
  statistics->finalizeStatistics();

  statistics->writeStatistics(grid);

  
}
