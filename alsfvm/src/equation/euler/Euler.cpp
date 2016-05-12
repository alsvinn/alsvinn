#include "alsfvm/equation/euler/Euler.hpp"

namespace alsfvm {
	namespace equation {
		namespace euler {
			const std::string Euler::name = "euler";

            const std::vector<std::string> Euler::conservedVariables = { "rho",
                                                                 "mx",
                                                                 "my",
                                                                 "mz",
                                                                 "E"};


            
            const std::vector<std::string> Euler::primitiveVariables = { "rho",
                                                                         "ux",
                                                                         "uy",
                                                                         "uz",
                                                                         "p"};

            const std::vector<std::string> Euler::extraVariables = { "p",
                                                                     "ux",
                                                                     "uy",
                                                                     "uz" };
		}
}
}
