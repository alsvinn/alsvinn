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
#include "alsfvm/diffusion/DiffusionOperator.hpp"
#include "alsfvm/volume/VolumeFactory.hpp"
#include "alsfvm/simulator/SimulatorParameters.hpp"
#include "alsfvm/reconstruction/tecno/ReconstructionFactory.hpp"

namespace alsfvm {
namespace diffusion {


///
/// Applies the Tecno diffusion to the operator. This will always take
/// the form
///
/// \f[R\Lambda R^{-1} \langle\langle v\rangle \rangle\f]
///
/// where \f$R\f$ is the matrix of eigenvalues of the flux jacobian, and
/// \f$\Lambda\f$ is either the Rusanov or Roe matrix. See
///
/// http://www.cscamm.umd.edu/tadmor/pub/TV+entropy/Fjordholm_Mishra_Tadmor_SINUM2012.pdf
///
/// The matrix \f$\Lambda\f$ is specified through the DiffusionMatrix template argument.
///
template<class Equation, template<class, int> class DiffusionMatrix>
class TecnoDiffusionCPU : public DiffusionOperator {
public:

    TecnoDiffusionCPU(volume::VolumeFactory& volumeFactory,
        alsfvm::shared_ptr<reconstruction::tecno::TecnoReconstruction> reconstruction,
        const simulator::SimulatorParameters& simulatorParameters);

    ///
    /// Applies numerical diffusion to the outputVolume given the data in conservedVolume.
    ///
    /// The numerical diffusion will be added to outputVolume, ie. the code will
    /// essentially work like
    ///
    /// \code
    /// outputVolume += diffusion(conservedVolume);
    /// \endcode
    ///
    ///
    virtual void applyDiffusion(volume::Volume& outputVolume,
        const volume::Volume& conservedVolume);

    ///
    /// Gets the total number of ghost cells this diffusion needs,
    /// this is typically governed by reconstruction algorithm.
    ///
    virtual size_t getNumberOfGhostCells() const;

private:
    alsfvm::volume::VolumeFactory volumeFactory;
    alsfvm::shared_ptr<reconstruction::tecno::TecnoReconstruction> reconstruction;

    // Reconstructed values (these are basically R^{-1}v,
    // where v is the entropy variables and R^{-1} is the inverse of the
    // eigenvalues of the flux.
    alsfvm::shared_ptr<volume::Volume> left;
    alsfvm::shared_ptr<volume::Volume> right;

    alsfvm::shared_ptr<volume::Volume> entropyVariablesLeft;
    alsfvm::shared_ptr<volume::Volume> entropyVariablesRight;

    Equation equation;
};
} // namespace diffusion
} // namespace alsfvm
