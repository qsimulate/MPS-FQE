import fqe
import pytest

from mps_fqe.wavefunction import MPSWavefunction, get_hf_mps


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
