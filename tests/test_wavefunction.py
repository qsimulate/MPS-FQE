import fqe
import numpy
import pytest

from mps_fqe.wavefunction import MPSWavefunction, get_hf_mps, get_random_mps
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

    mps = get_hf_mps(nele, sz, norb, bdim=50)
    out = mps.to_fqe_wavefunction()

    # only check up to a global phase
    assert abs(abs(fqe.vdot(out, wfn)) - 1) < tol


@pytest.mark.parametrize("nele, sz, norb", [
    (2, 0, 2), (2, 0, 4),
    (1, 1, 2), (1, -1, 2),
    (3, 1, 3), (3, -1, 3)
])
def test_random_wavefunction(nele, sz, norb):
    """Test that a random MPS is created with the correct particle-number
    and spin symmetry labels.
    """
    mps = get_random_mps(nele, sz, norb, bdim=50)
    wfn = mps.to_fqe_wavefunction()

    assert wfn.conserve_number()
    assert wfn.conserve_spin()

    sectors = list(wfn.sectors())
    sector = sectors[0]
    assert len(sectors) == 1
    assert sector[0] == nele
    assert sector[1] == sz


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


def test_expectationValue():
    nele, sz, norb = 4, 0, 4
    tol = 1E-13
    numpy.random.seed(77)

    # random wavefunction
    wfn = fqe.Wavefunction([[nele, sz, norb]])
    wfn.set_wfn(strategy="random")
    mps = MPSWavefunction.from_fqe_wavefunction(wfn, max_bond_dim=200)

    # random bra wavefunction
    bra_wfn = fqe.Wavefunction([[nele, sz, norb]])
    bra_wfn.set_wfn(strategy="random")
    bra_mps = MPSWavefunction.from_fqe_wavefunction(bra_wfn, max_bond_dim=200)

    # get random 1-particle restricted Hamiltonian
    h1 = numpy.random.rand(norb, norb)
    h1 += h1.transpose()
    hamiltonian = fqe.get_restricted_hamiltonian((h1,), e_0=0)

    ref_energy = wfn.expectationValue(hamiltonian)
    mps_energy = mps.expectationValue(hamiltonian)
    assert abs(ref_energy - mps_energy) < tol

    ref_de = wfn.expectationValue(hamiltonian, brawfn=bra_wfn)
    mps_de = mps.expectationValue(hamiltonian, brawfn=bra_mps)
    assert abs(ref_de - mps_de) < tol
