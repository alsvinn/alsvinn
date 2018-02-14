#include <gtest/gtest.h>
#include "alsuq/distribution/DistributionFactory.hpp"
#include "alsuq/generator/GeneratorFactory.hpp"
#include "utils/polyfit.hpp"

using namespace alsuq;

namespace {
struct MeanVarTestParameters {

    MeanVarTestParameters(const std::string& generatorName,
        const std::string& distName,
        real mean, real variance,
        std::vector<std::string> parameterNames,
        std::vector<real> parameterValues,
        int D)
        : generatorName(generatorName), distributionName(distName), mean(mean),
          variance(variance), D(D) {
        for (int i = 0; i < parameterNames.size(); ++i) {
            parameters.setParameter(parameterNames[i], parameterValues[i]);
        }
    }

    std::string generatorName;
    std::string distributionName;
    boost::property_tree::ptree ptree;
    alsuq::distribution::Parameters parameters{ptree};
    real mean;
    real variance;
    int D;
};

std::ostream& operator<<(std::ostream& os,
    const MeanVarTestParameters& parameters) {
    os << "\n{\n\tgeneratorName = " << parameters.generatorName
        << "\n\tdistributionName = " << parameters.distributionName
        << "\n\tmean = " << parameters.mean
        << "\n\tvariance = " << parameters.variance
        << std::endl << "}" << std::endl;
    return os;
}
}

struct MeanVarTest : public ::testing::TestWithParam <MeanVarTestParameters> {


    MeanVarTest () {

    }



    std::shared_ptr<alsuq::generator::Generator> makeGenerator(int M, int D) {
        auto parameters = GetParam();
        alsuq::generator::GeneratorFactory factory;
        return factory.makeGenerator(parameters.generatorName, D, M);
    }

    std::shared_ptr<alsuq::distribution::Distribution> makeDistribution(int M,
        int D) {
        auto parameters = GetParam();
        alsuq::distribution::DistributionFactory factory;
        return factory.createDistribution(parameters.distributionName, D, M,
                parameters.parameters);
    }
};



TEST_P(MeanVarTest, MeanVarTest) {
    int D = GetParam().D;
    // make sure we converge towards the mean with rate 1/2
    std::vector<real> logM;
    std::vector<real> logMeanError;
    std::vector<real> logVarError;

    const real actualVar = GetParam().variance;
    const real actualMean = GetParam().mean;

    for (int k = 4; k < 20; k++) {
        int M = 1 << k;
        auto generator = makeGenerator(M, D);
        auto dist = makeDistribution(M, D);
        std::vector<real> mean(40, 0);
        std::vector<real> M2(40, 0);
        int N = 0;

        for (int n = 0; n < M; ++n) {
            N += 1;

            for (int i = 0; i < D; ++i) {
                real x = dist->generate(*generator, i);
                real delta = x - mean[i];
                mean[i] += delta / N;
                real delta2 = x - mean[i];
                M2[i] += delta * delta2;
            }
        }


        std::vector<real> var(D);

        real errorMean = 0.0;
        real errorVar = 0.0;

        for (int i = 0; i < D; ++i ) {
            var[i] = M2[i] / (N - 1);
            errorMean += std::pow(std::abs(mean[i] - actualMean), 2);
            errorVar += std::pow(std::abs(var[i] - actualVar), 2);

        }

        logM.push_back(std::log(M));
        logMeanError.push_back(std::log(std::sqrt(errorMean)));
        logVarError.push_back(std::log(std::sqrt(errorVar)));
    }

    double rateMean = alsfvm::linearFit(logM, logMeanError)[0];

    ASSERT_LE(0.49, -rateMean);

    double rateVar = alsfvm::linearFit(logM, logVarError)[0];

    ASSERT_LE(0.49, -rateVar);
}

INSTANTIATE_TEST_CASE_P(MeanVarConvergenceTests,
    MeanVarTest,
    ::testing::Values(
        MeanVarTestParameters("well512a", "normal", 0, 1, {"mean", "sd"}, {0, 1}, 40),
        MeanVarTestParameters("well512a", "uniform", 0.5, 1.0 / 12., {"lower", "upper"}, {0, 1},
            40),
        MeanVarTestParameters("stlmersenne", "normal", 0, 1, {"mean", "sd"}, {0, 1},
            40),
        MeanVarTestParameters("stlmersenne", "uniform", 0.5, 1.0 / 12., {"lower", "upper"}, {0, 1},
            40)

    ));
