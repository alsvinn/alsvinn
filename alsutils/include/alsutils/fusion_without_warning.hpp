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
