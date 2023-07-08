import fqe
import numpy
import pytest

from mps_fqe.wavefunction import MPSWavefunction, get_hf_mps
from mps_fqe.hamiltonian import mpo_from_fqe_hamiltonian


@pytest.mark.parametrize("params", [
    [[3, 1, 3]],
    [[2, 0, 2], [3, 1, 2]]])
def test_wavefunction_vs_fqe(params):
    tol = 1e-14
    wfn = fqe.Wavefunction(params)
    wfn.set_wfn(strategy="random")

    mps = MPSWavefunction.from_fqe_wavefunction(wfn)
    out = mps.to_fqe_wavefunction()

    # only check up to a global phase
    assert abs(abs(fqe.vdot(out, wfn)) - 1) < tol


@pytest.mark.parametrize("nele, sz, norb", [
    (2, 0, 2), (2, 0, 4),  # only doubly occupied orbitals (RHF)
    (1, 1, 2), (1, -1, 2),  # only singly occupied orbitals (single spin)
    (3, 1, 3), (3, -1, 3)  # doubly and singly occupied (ROHF)
])
def test_hf_wavefunction(nele, sz, norb):
    tol = 1e-14

    wfn = fqe.Wavefunction([[nele, sz, norb]])
    wfn.set_wfn(strategy="hartree-fock")
    # wfn.print_wfn()

    mps = get_hf_mps(nele, sz, norb, bdim=50)
    out = mps.to_fqe_wavefunction()
    # out.print_wfn()

    # only check up to a global phase
    assert abs(abs(fqe.vdot(out, wfn)) - 1) < tol


def test_wavefunction_apply_diagonal():
    nele, sz, norb = 4, 0, 4
    numpy.random.seed(77)
    wfn = fqe.Wavefunction([[nele, sz, norb]])
    wfn.set_wfn(strategy="random")
    mps = MPSWavefunction.from_fqe_wavefunction(wfn, max_bond_dim=200)

    terms = numpy.random.rand(norb)
    hamiltonian = fqe.get_diagonal_hamiltonian(terms,
                                               e_0=-1.0)
    mpo = mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)
    ref = mps.apply(mpo)
    out = mps.apply_linear(mpo)

    # check the 1-rdm
    assert numpy.allclose(out.rdm('i^ j'), ref.rdm('i^ j'))


def test_wavefunction_apply_1particle():
    nele, sz, norb = 4, 0, 4
    numpy.random.seed(77)
    wfn = fqe.Wavefunction([[nele, sz, norb]])
    wfn.set_wfn(strategy="random")
    mps = MPSWavefunction.from_fqe_wavefunction(wfn, max_bond_dim=200)

    h1 = numpy.random.rand(norb, norb)
    h1 += h1.transpose()
    hamiltonian = fqe.get_restricted_hamiltonian((h1,), e_0=0)
    mpo = mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)
    ref = mps.apply(mpo)
    out = mps.apply_linear(mpo)

    # check the 1-rdm
    assert numpy.allclose(out.rdm('i^ j'), ref.rdm('i^ j'))
