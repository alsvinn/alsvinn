#include "alsfvm/equation/euler/Euler.hpp"

namespace alsfvm {
	namespace equation {
		namespace euler {
			const std::string Euler<1>::name = "euler1";
            const std::string Euler<2>::name = "euler2";
            const std::string Euler<3>::name = "euler3";

            const std::vector<std::string> Euler<3>::conservedVariables = { "rho",
                                                                 "mx",
                                                                 "my",
                                                                 "mz",
                                                                 "E"};


            
            const std::vector<std::string> Euler<3>::primitiveVariables = { "rho",
                                                                         "ux",
                                                                         "uy",
                                                                         "uz",
                                                                         "p"};

            const std::vector<std::string> Euler<3>::extraVariables = { "p",
                                                                     "ux",
                                                                     "uy",
                                                                     "uz" };


            const std::vector<std::string> Euler<2>::conservedVariables = { "rho",
                "mx",
                "my",
                "E" };



            const std::vector<std::string> Euler<2>::primitiveVariables = { "rho",
                "ux",
                "uy",
                "p" };

            const std::vector<std::string> Euler<2>::extraVariables = { "p",
                "ux",
                "uy"
                 };


            const std::vector<std::string> Euler<1>::conservedVariables = { "rho",
                "mx",
                "E" };



            const std::vector<std::string> Euler<1>::primitiveVariables = { "rho",
                "ux",
                "p" };

            const std::vector<std::string> Euler<1>::extraVariables = { "p",
                "ux"
            };



		}
}
}
