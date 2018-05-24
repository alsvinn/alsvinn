/* Copyright (c) 2018 ETH Zurich, Kjetil Olsen Lye
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

#pragma once
#include "alsfvm/functional/FunctionalFactory.hpp"
#include <boost/preprocessor.hpp>
#define REGISTER_FUNCTIONAL(platform, name, classname) namespace { \
    static struct BOOST_PP_CAT(RegisterStruct, BOOST_PP_CAT(platform, name)) { \
        BOOST_PP_CAT(RegisterStruct, BOOST_PP_CAT(platform, name))() { \
            alsfvm::functional::FunctionalFactory::registerFunctional(#platform, #name, [](const alsfvm::functional::Functional::Parameters& params) { \
                std::shared_ptr<alsfvm::functional::Functional> functional; \
                functional.reset(new classname(params)); \
                return functional; \
                });\
            } \
        } BOOST_PP_CAT(BOOST_PP_CAT(registerObject, platform),name); \
    }
