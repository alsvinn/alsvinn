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
#include "alsfvm/types.hpp"
#include "alsfvm/grid/Grid.hpp"
#include "alsfvm/mpi/CellExchanger.hpp"

namespace alsfvm {
namespace mpi {
namespace domain {

//! Contains information about the domain this processor is assigned, as well
//! as the given neighbours
class DomainInformation {
public:
    DomainInformation(alsfvm::shared_ptr<grid::Grid> grid,
        CellExchangerPtr cellExchanger);

    alsfvm::shared_ptr<grid::Grid> getGrid();

    CellExchangerPtr getCellExchanger();
private:
    alsfvm::shared_ptr<grid::Grid> grid;
    CellExchangerPtr cellExchanger;

};

typedef alsfvm::shared_ptr<DomainInformation> DomainInformationPtr;
} // namespace domain
} // namespace mpi
} // namespace alsfvm
