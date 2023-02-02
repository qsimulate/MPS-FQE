import os
import numpy as np
import fqe
import numpy
import pytest
import itertools
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from openfermion.chem import make_atomic_ring

from mps_fqe.wavefunction import MPSWavefunction
from mps_fqe.hamiltonian import MPOHamiltonian


def get_H_ring_data(amount_H):
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Hring_{amount_H}.hdf5")
    molecule = make_atomic_ring(amount_H, 1.0, "sto-3g", atom_type="H", charge=0, filename=filename)

    if os.path.isfile(filename):
        molecule.load()

    if molecule.hf_energy is None:
        molecule = generate_H_ring_data(molecule)
    return molecule


def generate_H_ring_data(molecule):
    from openfermionpyscf import run_pyscf

    molecule = run_pyscf(molecule, run_scf=True)
    molecule.save()
    return molecule

def test_H_ring_energy(amount_H):
    molecule = get_H_ring_data(amount_H)
    nele = molecule.n_electrons
    sz = molecule.multiplicity - 1
    norbs = molecule.n_orbitals
    h1, h2 = molecule.get_integrals()
    
    fqe_wfn = fqe.Wavefunction([[nele, sz, norbs]])
    fqe_wfn.set_wfn(strategy="hartree-fock")
    fqe_wfn.normalize()

    hamiltonian = fqe.get_restricted_hamiltonian((h1, numpy.einsum("ijlk", -0.5*h2)), e_0=molecule.nuclear_repulsion)

    MPO1 = MPOHamiltonian.from_fqe_specific(fqe_wfn=fqe_wfn,
                                            fqe_ham=hamiltonian,
                                            flat=True,)
    #Curiously missing a site from this instance
    MPO2 = MPOHamiltonian.from_fqe_hamiltonian(fqe_wfn=fqe_wfn,
                                               fqe_ham=hamiltonian,
                                               flat=True,)
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn)
    print(fqe_wfn.expectationValue(hamiltonian))
    print(mps.expectationValue(MPO1))
    print(mps.expectationValue(MPO2))
    print(molecule.hf_energy)

if __name__ == '__main__':
    for nH in range(2, 7):
        test_H_ring_energy(nH)
