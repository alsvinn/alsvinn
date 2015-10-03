#pragma once

#include <sstream>
#include <exception>
#include <stdexcept>
#include <boost/current_function.hpp>

/// 
/// Throws an exception with the given message
///
#define THROW(message) {\
	std::stringstream ssForException; \
	ssForException << message; \
	ssForException << std::endl << "At " << __FILE__<<":" << __LINE__ << std::endl;\
	ssForException << std::endl << "In function: " << BOOST_CURRENT_FUNCTION << std::endl;\
	throw std::runtime_error(ssForException.str());\
}
	
