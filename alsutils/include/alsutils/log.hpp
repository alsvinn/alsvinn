#pragma once

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#define ALSVINN_LOG(severity, message) BOOST_LOG_TRIVIAL(severity) << message;
