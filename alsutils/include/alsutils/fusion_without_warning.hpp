#pragma once
//! The only role of this file is to include boost fusion with a single warning
//!
//! See https://stackoverflow.com/questions/6321839/how-to-disable-warnings-for-particular-include-files/6321977#6321977
//! Supported compilers: gcc
//! It seems it is not needed anymore, therefore disabling. It does give warnings,
//! but only on NVCC, which we can not disable anyway:
//#ifdef __GNUC__
//    #pragma GCC system_header
//#endif
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/container.hpp>
#include <boost/fusion/sequence/intrinsic/at_key.hpp>
