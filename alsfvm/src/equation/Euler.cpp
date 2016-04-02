#include "alsfvm/equation/euler/Euler.hpp"

namespace alsfvm {
	namespace equation {
		namespace euler {
			const std::string Euler::name = "euler";

            const std::vector<std::string> conservedVariables = {"rho",
                                                                 "mx",
                                                                 "my",
                                                                 "mz",
                                                                 "E"};


            static const std::vector<std::string> primitiveVariables = { "rho",
                                                                         "ux",
                                                                         "uy",
                                                                         "uz",
                                                                         "p"};

            static const std::vector<std::string> extraVariables = { "p",
                                                                     "ux",
                                                                     "uy",
                                                                     "uz" };
		}
}
}
