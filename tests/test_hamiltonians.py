import os
import numpy
from numpy import einsum

import pytest
import fqe
from openfermion import FermionOperator
from openfermion.chem import make_atomic_ring
from mps_fqe.wavefunction import MPSWavefunction
from mps_fqe.hamiltonian import MPOHamiltonian


def get_H_ring_data(amount_H):
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            f"Hring_{amount_H}.hdf5")
    molecule = make_atomic_ring(amount_H, 1.0, "sto-3g", atom_type="H",
                                charge=0, filename=filename)

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


@pytest.mark.parametrize("amount_H", range(2, 5))
def test_restricted(amount_H):
    molecule = get_H_ring_data(amount_H)
    nele = molecule.n_electrons
    sz = molecule.multiplicity - 1
    norbs = molecule.n_orbitals
    h1, h2 = molecule.get_integrals()

    fqe_wfn = fqe.Wavefunction([[nele, sz, norbs]])
    fqe_wfn.set_wfn(strategy="hartree-fock")
    fqe_wfn.normalize()
    e_0 = molecule.nuclear_repulsion
    hamiltonian = fqe.get_restricted_hamiltonian((h1, einsum("ijlk", -0.5*h2)),
                                                 e_0=e_0)

    mpo = MPOHamiltonian.from_fqe_hamiltonian(fqe_wfn=fqe_wfn,
                                              fqe_ham=hamiltonian,
                                              flat=True,)
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn)

    assert numpy.isclose(mps.expectationValue(mpo),
                         fqe_wfn.expectationValue(hamiltonian),
                         atol=1e-12)


@pytest.mark.parametrize("n_electrons", [3, 5])
def test_diagonal_coulomb(n_electrons, sz=1, n_orbitals=4):
    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    vij = numpy.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals),
                      dtype=numpy.complex128)
    for i in range(n_orbitals):
        for j in range(n_orbitals):
            vij[i, j] += 4*(i % n_orbitals + 1)*(j % n_orbitals + 1)*0.21

    fqe_wfn.set_wfn(strategy='random')
    hamiltonian = fqe.get_diagonalcoulomb_hamiltonian(vij,
                                                      e_0=numpy.random.rand())
    mpo = MPOHamiltonian.from_fqe_hamiltonian(fqe_wfn=fqe_wfn,
                                              fqe_ham=hamiltonian,
                                              flat=True)
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)
    assert numpy.isclose(mps.expectationValue(mpo),
                         fqe_wfn.expectationValue(hamiltonian),
                         atol=1e-12)


@pytest.mark.parametrize("n_electrons,sz,n_orbitals", [(2, 2, 4),
                                                       (6, 2, 4),
                                                       (4, 0, 4)])
def test_diagonal_hamiltonian(n_electrons, sz, n_orbitals):
    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    fqe_wfn.set_wfn(strategy='random')
    terms = numpy.random.rand(n_orbitals)
    hamiltonian = fqe.get_diagonal_hamiltonian(terms,
                                               e_0=numpy.random.rand())
    mpo = MPOHamiltonian.from_fqe_hamiltonian(fqe_wfn=fqe_wfn,
                                              fqe_ham=hamiltonian,
                                              flat=True)
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)
    assert numpy.isclose(mps.expectationValue(mpo),
                         fqe_wfn.expectationValue(hamiltonian),
                         atol=1e-12)


@pytest.mark.parametrize("n_electrons,sz,n_orbitals", [(2, 2, 4),
                                                       (6, 2, 4),
                                                       (4, 0, 4)])
def test_sparse_hamiltonian(n_electrons, sz, n_orbitals):
    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    fqe_wfn.set_wfn(strategy='random')
    coeffs = numpy.random.rand(4)
    operator = coeffs[0] * FermionOperator('0^ 0') \
        + coeffs[1] * FermionOperator('0^ 2 5^ 7') \
        + coeffs[2] * FermionOperator('6 0^') \
        + coeffs[3] * FermionOperator('3^ 3')

    e_0 = numpy.random.rand()
    hamiltonian = fqe.sparse_hamiltonian.SparseHamiltonian(operator,
                                                           e_0=e_0)
    mpo = MPOHamiltonian.from_fqe_hamiltonian(fqe_wfn=fqe_wfn,
                                              fqe_ham=hamiltonian,
                                              flat=True)
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)
    assert numpy.isclose(mps.expectationValue(mpo),
                         fqe_wfn.expectationValue(hamiltonian),
                         atol=1e-12)
