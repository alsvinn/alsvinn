#include <gtest/gtest.h>

#include "alsfvm/equation/CellComputerFactory.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"


using namespace alsfvm;
using namespace alsfvm::memory;
using namespace alsfvm::equation;
using namespace alsfvm::volume;

class TestExtraComputation : public ::testing::Test {
public:
    std::shared_ptr<DeviceConfiguration> deviceConfiguration;
    const std::string equation;
    const std::string platform;
    const size_t nx;
    const size_t ny;
    const size_t nz;
    std::shared_ptr<MemoryFactory> memoryFactory;
    VolumeFactory volumeFactory;
    std::shared_ptr<Volume> conservedVolume;
    std::shared_ptr<Volume> extraVolume;
    CellComputerFactory cellComputerFactory;
    TestExtraComputation()
        : deviceConfiguration(new DeviceConfiguration),
          equation("euler"),
          platform("cpu"),
          nx(10), ny(10), nz(10),
          memoryFactory(new MemoryFactory("HostMemory", deviceConfiguration)),
          volumeFactory("euler", memoryFactory),
          conservedVolume(volumeFactory.createConservedVolume(nx, ny, nz)),
          extraVolume(volumeFactory.createExtraVolume(nx, ny, nz)),
          cellComputerFactory(platform, equation)
    {

    }
};

TEST_F(TestExtraComputation, ConstructTest) {
    auto cellComputer =cellComputerFactory.createComputer();
}

TEST_F(TestExtraComputation, CheckSimpleValue) {
    auto cellComputer =cellComputerFactory.createComputer();


}
