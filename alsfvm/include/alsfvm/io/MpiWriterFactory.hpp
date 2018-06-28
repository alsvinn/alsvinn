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
#include "alsfvm/io/WriterFactory.hpp"
#include "alsfvm/mpi/Configuration.hpp"

namespace alsfvm {
namespace io {

//! This implements the abstract factory pattern
//! The reason for doing this is that we sometimes want to use
//! mpi writers, and sometimes not.
class MpiWriterFactory : public WriterFactory {
public:
    MpiWriterFactory(mpi::ConfigurationPtr configuration);

    virtual alsfvm::shared_ptr<Writer> createWriter(const std::string& name,
        const std::string& baseFilename, const Parameters& parameters) override;

private:
    mpi::ConfigurationPtr configuration;
};
} // namespace io
} // namespace alsfvm
