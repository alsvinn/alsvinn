#pragma once
#include <vector>
#include <string>
#include <map>
#include "alsfvm/types.hpp"
namespace alsfvm { namespace init { 

    //! Parameters for the initial data. 
    //! These are typically used to give random
    //! inputs.
    class Parameters {
    public:

        //! Add a a parameter to the parameters.
        void addParameter(const std::string& name, const std::vector<real>& value);


        std::vector<std::string> getParameterNames() const;

        //! Each parameter is represented by an array 
        //! A scalar is then represented by a length one array.
        const std::vector<real>& getParameter(const std::string& name) const;

    private:
        std::map<std::string,  std::vector<real> > parameters;
    };
} // namespace init
} // namespace alsfvm
