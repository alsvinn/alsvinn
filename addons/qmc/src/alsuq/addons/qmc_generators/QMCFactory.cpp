#include "alsuq/addons/qmc_generators/QMCDistribution.hpp"
#include "alsuq/addons/qmc_generators/QMCFactory.hpp"
#include <alsutils/base/Factory.hpp>
#include <alsutils/config.hpp>
#include <alsutils/error/Exception.hpp>
#include <alsutils/log.hpp>
namespace alsuq {
namespace addons {
namespace qmc_generators {

namespace {
using QMCFactoryImpl =
    alsutils::base::Factory<QMCDistribution, const size_t, const size_t, const alsutils::parameters::Parameters>;
}

bool QMCFactory::addDistribution(
    const std::string& name,
    const std::function<alsfvm::shared_ptr<alsuq::addons::qmc_generators::QMCDistribution> ( const size_t, const size_t, const alsutils::parameters::Parameters&)>
    & creator) {

    return QMCFactoryImpl::registerClass(name, creator);


}

alsfvm::shared_ptr<QMCDistribution> QMCFactory::makeQMCDistribution(
    const std::string& name,
    const size_t dimensions,
    const size_t numberVariables,
    const alsutils::parameters::Parameters& parameters) {
#ifdef ALSVINN_BUILD_QMC
    return QMCFactoryImpl::createInstance(name, dimensions, numberVariables,
            parameters);
#else
    THROW("Alsvinn was build without qmc support, enable it by adding \"-DALSVINN_BUILD_QMC=ON\" when running cmake.");
#endif
}

}
}
}
