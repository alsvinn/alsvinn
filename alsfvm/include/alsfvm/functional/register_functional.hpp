#pragma once
#include "alsfvm/functional/FunctionalFactory.hpp"
#include <boost/preprocessor.hpp>
#define REGISTER_FUNCTIONAL(platform, name, classname) namespace { \
    static struct BOOST_PP_CAT(RegisterStruct, BOOST_PP_CAT(platform, name)) { \
        BOOST_PP_CAT(RegisterStruct, BOOST_PP_CAT(platform, name))() { \
            alsfvm::functional::FuncationalFactory::registerFunctional(#platform, #name, [](const alsfvm::functional::Functional::Parameters& params) { \
                std::shared_ptr<alsfvm::functional::Functional> functional; \
                functional.reset(new classname(params)); \
                return functional; \
                });\
            } \
        } BOOST_PP_CAT(BOOST_PP_CAT(registerObject, platform),name); \
    }
