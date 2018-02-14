#pragma once
#include "alsuq/types.hpp"
#include <boost/noncopyable.hpp>

namespace alsuq {
namespace generator {

//! A generator is used to generate random numbers,
//! use a distribution to get them under some distribution
//!
//! \note All generators are singletons
class Generator : public boost::noncopyable {
public:
    virtual ~Generator() {};

    //! Generates a uniformly distributed number between 0 and 1
    virtual real generate(size_t component) = 0;
};
} // namespace generator
} // namespace alsuq
