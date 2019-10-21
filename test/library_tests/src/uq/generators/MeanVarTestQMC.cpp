/* Copyright (c) 2019 ETH Zurich, Kjetil Olsen Lye
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <gtest/gtest.h>
#include "alsuq/distribution/DistributionFactory.hpp"
#include "alsuq/generator/GeneratorFactory.hpp"
#include "utils/polyfit.hpp"
#include "alsutils/config.hpp"

using namespace alsuq;
#ifdef ALSVINN_BUILD_QMC
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
        for (size_t i = 0; i < parameterNames.size(); ++i) {
            parameters.addDoubleParameter(parameterNames[i], parameterValues[i]);
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

struct QMCMeanVarTest : public ::testing::TestWithParam
    <MeanVarTestParameters> {


    QMCMeanVarTest () {

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



TEST_P(QMCMeanVarTest, QMCMeanVarTest) {
    int D = GetParam().D;
    // make sure we converge towards the mean with rate 1/2
    std::vector<real> logM;
    std::vector<real> logMeanError;
    std::vector<real> logVarError;

    const real actualVar = GetParam().variance;
    const real actualMean = GetParam().mean;

    for (int k = 4; k < 13; k++) {
        int M = 1 << k;
        auto generator = makeGenerator(M, D);
        auto dist = makeDistribution(M, D);
        std::vector<real> mean(40, 0);
        std::vector<real> M2(40, 0);
        int N = 0;

        for (int n = 0; n < M; ++n) {
            N += 1;

            for (int i = 0; i < D; ++i) {
                real x = dist->generate(*generator, i, n);
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

    std::cout << "Rate " << rateMean << ", " << rateVar << std::endl;
}

INSTANTIATE_TEST_CASE_P(QMCMeanVarConvergenceTests,
    QMCMeanVarTest,
    ::testing::Values(

        MeanVarTestParameters("stlmersenne", "qmc_sobol", 0.5, 1.0 / 12., {"lower", "upper"}, {0, 1},
            40), MeanVarTestParameters("stlmersenne", "qmc_halton", 0.5, 1.0 / 12., {"lower", "upper"}, {0, 1},
            40), MeanVarTestParameters("stlmersenne", "qmc_halton409", 0.5, 1.0 / 12., {"lower", "upper"}, {0, 1},
            40), MeanVarTestParameters("stlmersenne", "qmc_faure", 0.5, 1.0 / 12., {"lower", "upper"}, {0, 1},
            40), MeanVarTestParameters("stlmersenne", "qmc_hammersley", 0.5, 1.0 / 12., {"lower", "upper"}, {0, 1},
            40), MeanVarTestParameters("stlmersenne", "qmc_latin_random", 0.5, 1.0 / 12., {"lower", "upper"}, {0, 1},
            40)

    ));

#endif
