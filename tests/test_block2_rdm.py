import numpy
import pytest
import fqe
from mps_fqe.wavefunction import MPSWavefunction
fqe.settings.use_accelerated_code = False


@pytest.mark.parametrize("n_electrons,sz,n_orbitals", [(2, 2, 4),
                                                       (6, -2, 4),
                                                       (5, 1, 4),
                                                       (6, 2, 4),
                                                       (4, 0, 6)])
def test_block2_rdm1(n_electrons, sz, n_orbitals):

    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    fqe_wfn.set_wfn(strategy='random')
    fqe_wfn.normalize()
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)

    rdm = mps.rdm("i^ j", block2=True)
    fqe_rdm = fqe_wfn.rdm("i^ j")

    assert numpy.allclose(rdm,
                          fqe_rdm,
                          atol=1E-12)


@pytest.mark.parametrize("n_electrons,sz,n_orbitals", [(3, 1, 4),
                                                       (6, -2, 4),
                                                       (5, 1, 4),
                                                       (6, 2, 4),
                                                       (4, 0, 6)])
def test_block2_rdm2(n_electrons, sz, n_orbitals):
    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    fqe_wfn.set_wfn(strategy='random')
    fqe_wfn.normalize()
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)

    rdm = mps.rdm("i^ j^ k l", block2=True)
    fqe_rdm = fqe_wfn.rdm("i^ j^ k l")
    assert numpy.allclose(rdm,
                          fqe_rdm,
                          atol=1E-12)


@pytest.mark.parametrize("n_electrons,sz,n_orbitals",  [(3, 1, 4),
                                                        (6, -2, 4),
                                                        (5, 1, 4),
                                                        (6, 2, 4),
                                                        (4, 0, 6)])
def test_block2_rdm3(n_electrons, sz, n_orbitals):
    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    fqe_wfn.set_wfn(strategy='random')
    fqe_wfn.normalize()
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)

    rdm = mps.rdm("i^ j^ k^ l m n", block2=True)
    fqe_rdm = fqe_wfn.rdm("i^ j^ k^ l m n")
    assert numpy.allclose(rdm,
                          fqe_rdm,
                          atol=1E-12)
