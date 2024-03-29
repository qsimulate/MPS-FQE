import numpy
from numpy import einsum

import pytest
import fqe
from openfermion import FermionOperator
from mps_fqe.wavefunction import MPSWavefunction
from mps_fqe.hamiltonian import mpo_from_fqe_hamiltonian
from .test_H_ring import get_H_ring_data


@pytest.mark.parametrize("amount_H", range(3, 5))
@pytest.mark.parametrize("complex_h", [False, True])
def test_restricted(amount_H, complex_h):
    molecule = get_H_ring_data(amount_H)
    nele = molecule.n_electrons
    sz = molecule.multiplicity - 1
    norbs = molecule.n_orbitals
    h1, h2 = molecule.get_integrals()
    if complex_h:
        h1 = h1.astype(numpy.complex128)

    fqe_wfn = fqe.Wavefunction([[nele, sz, norbs]])
    fqe_wfn.set_wfn(strategy="hartree-fock")
    fqe_wfn.normalize()
    e_0 = molecule.nuclear_repulsion
    hamiltonian = fqe.get_restricted_hamiltonian((h1,
                                                  einsum("ijlk", -0.5 * h2)),
                                                 e_0=e_0)

    mpo = mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn)

    assert numpy.isclose(mps.expectationValue(mpo),
                         fqe_wfn.expectationValue(hamiltonian),
                         atol=1e-12)


@pytest.mark.parametrize("n_electrons", [3, 5])
@pytest.mark.parametrize("full", [True, False])
def test_diagonal_coulomb(n_electrons, full, sz=1, n_orbitals=4):
    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    if full:
        vij = numpy.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals),
                          dtype=numpy.complex128)
        for i in range(n_orbitals):
            for j in range(n_orbitals):
                vij[i, j] += 4*(i % n_orbitals + 1)*(j % n_orbitals + 1)*0.21

    else:
        vij = numpy.zeros((n_orbitals, n_orbitals), dtype=numpy.complex128)
        for i in range(n_orbitals):
            for j in range(n_orbitals):
                vij[i, j] += 4*(i % n_orbitals + 1)*(j % n_orbitals + 1)*0.21

    fqe_wfn.set_wfn(strategy='random')
    hamiltonian = fqe.get_diagonalcoulomb_hamiltonian(vij,
                                                      e_0=numpy.random.rand())
    mpo = mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)
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
    mpo = mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)
    assert numpy.isclose(mps.expectationValue(mpo),
                         fqe_wfn.expectationValue(hamiltonian),
                         atol=1e-12)


@pytest.mark.parametrize("n_electrons,sz,n_orbitals", [(2, 2, 4),
                                                       (6, 2, 4),
                                                       (4, 0, 6)])
def test_sparse_hamiltonian(n_electrons, sz, n_orbitals):
    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    fqe_wfn.set_wfn(strategy='random')
    coeffs = numpy.random.rand(4)
    operator = coeffs[0] * FermionOperator('0^ 0') \
        + coeffs[0] * FermionOperator('1^ 3') \
        + coeffs[1] * FermionOperator('0^ 2 5^ 7') \
        + coeffs[2] * FermionOperator('6 0^') \
        + coeffs[3] * FermionOperator('3^ 3')

    e_0 = numpy.random.rand()
    hamiltonian = fqe.sparse_hamiltonian.SparseHamiltonian(operator,
                                                           e_0=e_0)
    mpo = mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian,
                                   n_sites=n_orbitals)
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)
    assert numpy.isclose(mps.expectationValue(mpo),
                         fqe_wfn.expectationValue(hamiltonian),
                         atol=1e-12)


def test_sparse_nsites_error():
    operator = FermionOperator('0^ 0')
    hamiltonian = fqe.sparse_hamiltonian.SparseHamiltonian(operator)
    err = "Must provide n_sites for sparse Hamiltonian"
    with pytest.raises(ValueError, match=err):
        mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)


def test_mpo_type_error():

    class FakeHamil:
        def e_0(self):
            return 0

        def dim(self):
            return 1

    hamiltonian = FakeHamil()
    err = "Have not implemented MPO for"
    with pytest.raises(TypeError, match=err):
        mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)


def test_restricted_mpo_value_error():
    n = 2
    h1 = numpy.zeros((n, n))
    h2 = numpy.zeros((n, n, n, n))
    h3 = numpy.zeros((n, n, n, n, n, n))
    hamiltonian = fqe.restricted_hamiltonian.RestrictedHamiltonian(
        tensors=(h1, h2, h3))
    err = "3-body or higher interactions are not supported"
    with pytest.raises(ValueError, match=err):
        mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)
