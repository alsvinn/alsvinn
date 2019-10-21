#include <alsuq/addons/qmc_generators/QMCDistribution.hpp>
namespace alsuq {
namespace addons {
namespace qmc_generators {

class QMCFactory {
public:
    static bool addDistribution(const std::string& name,
        const std::function<alsfvm::shared_ptr<QMCDistribution>(const size_t,
            const size_t, const alsutils::parameters::Parameters&)>& creator);

    static alsfvm::shared_ptr<QMCDistribution> makeQMCDistribution(
        const std::string& name,
        const size_t dimensions,
        const size_t numberVariables,
        const alsutils::parameters::Parameters& parameters);
};

}
}
}
