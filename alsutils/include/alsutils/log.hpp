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
#include <sstream>
#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/sources/logger.hpp>

#define INFO alsutils::log::severity_level::info
#define NOTIFICATION alsutils::log::severity_level::notification
#define WARNING alsutils::log::severity_level::warning
#define ERROR alsutils::log::severity_level::error
#define CRITICAL alsutils::log::severity_level::critical



#define ALSVINN_LOG(severity, message) { \
    std::stringstream ssForLog; \
    ssForLog << message; \
    BOOST_LOG_SEV(::alsutils::log::alsvinnLogger::get(), severity) << ssForLog.str(); \
    }

namespace alsutils {
namespace log {

enum severity_level {
    info,
    notification,
    warning,
    error,
    critical
};


BOOST_LOG_INLINE_GLOBAL_LOGGER_DEFAULT(alsvinnLogger,
    boost::log::sources::severity_logger_mt<severity_level>)


void inline setLogFile(const std::string& filename) {
    boost::log::add_file_log(filename, boost::log::keywords::auto_flush = true );
}
}
}
