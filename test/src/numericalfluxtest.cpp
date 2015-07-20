#include <gtest/gtest.h>

#include "alsfvm/types.hpp"
#include "alsfvm/numflux/NumericalFluxFactory.hpp"
#include "alsfvm/numflux/euler/NumericalFluxCPU.hpp"
#include "alsfvm/numflux/euler/HLL.hpp"

using namespace alsfvm::numflux;
using namespace alsfvm;
using namespace alsfvm::volume;

class NumericalFluxTest : public ::testing::Test {
public:
    std::string equation;
    std::string flux;
    std::string reconstruction;
    std::shared_ptr<DeviceConfiguration> deviceConfiguration;
    NumericalFluxFactory fluxFactory;
    grid::Grid grid;
    NumericalFluxTest()
        : equation("euler"), flux("HLL"), reconstruction("none"),
          deviceConfiguration(new DeviceConfiguration),
          fluxFactory(equation, flux, reconstruction, deviceConfiguration),
          grid(rvec3(0,0,0), rvec3(1,1,1), ivec3(20, 20, 20))
    {

    }
};

TEST_F(NumericalFluxTest, ConstructionTest) {
    auto numericalFlux = fluxFactory.createNumericalFlux(grid);
}
