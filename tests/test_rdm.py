import numpy
import pytest
import fqe
from mps_fqe.wavefunction import MPSWavefunction
fqe.settings.use_accelerated_code = False


@pytest.mark.parametrize("n_electrons,sz,n_orbitals", [(6, -2, 4),
                                                       (6, 2, 4),
                                                       (2, 2, 4),
                                                       (5, 1, 4),
                                                       (4, 0, 6)])
def test_rdm1(n_electrons, sz, n_orbitals):
    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    fqe_wfn.set_wfn(strategy='random')
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)
    # Spatial orbitals
    assert numpy.allclose(mps.rdm('i^ j', block2=False),
                          fqe_wfn.rdm('i^ j'),
                          atol=1e-12)
    # Spin orbitals
    assert numpy.isclose(mps.rdm('0^ 2', block2=False),
                         fqe_wfn.rdm('0^ 2'),
                         atol=1e-12)

    assert numpy.isclose(mps.rdm('1^ 3', block2=False),
                         fqe_wfn.rdm('1^ 3'),
                         atol=1e-12)


@pytest.mark.parametrize("n_electrons,sz,n_orbitals", [(6, -2, 4),
                                                       (6, 2, 4),
                                                       (2, 2, 4),
                                                       (5, 1, 4),
                                                       (4, 0, 6)])
def test_rdm2(n_electrons, sz, n_orbitals):
    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    fqe_wfn.set_wfn(strategy='random')
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)

    assert numpy.isclose(mps.rdm('0^ 2^ 0 4', block2=False),
                         fqe_wfn.rdm('0^ 2^ 0 4'),
                         atol=1e-12)

    assert numpy.isclose(mps.rdm('1^ 3^ 3 5', block2=False),
                         fqe_wfn.rdm('1^ 3^ 3 5'),
                         atol=1e-12)

    assert numpy.allclose(mps.rdm('i^ j^ k l', block2=False),
                          fqe_wfn.rdm('i^ j^ k l'),
                          atol=1E-12)


@pytest.mark.parametrize("n_electrons,sz,n_orbitals", [(3, 1, 4)])
def test_rdm3(n_electrons, sz, n_orbitals):
    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    fqe_wfn.set_wfn(strategy='random')
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)

    assert numpy.isclose(mps.rdm('0^ 2^ 3^ 0 4 1', block2=False),
                         fqe_wfn.rdm('0^ 2^ 3^ 0 4 1'),
                         atol=1e-12)

    assert numpy.allclose(mps.rdm('i^ j^ k^ l m n', block2=False),
                          fqe_wfn.rdm('i^ j^ k^ l m n'),
                          atol=1E-12)


def test_rdm_exceptions():
    n_electrons = 6
    sz = 2
    n_orbitals = 4
    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    fqe_wfn.set_wfn(strategy='random')
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)
    err = "RDM must have even number of operators."
    with pytest.raises(ValueError, match=err):
        _ = mps.rdm('0^')

    err = "Transition density not implemented with block2 driver."
    with pytest.raises(ValueError, match=err):
        _ = mps.rdm('i^ j', block2=True, brawfn=mps)

    err = "RDM is only implemented up to 3pdm."
    with pytest.raises(ValueError, match=err):
        _ = mps.rdm('i^ j^ k^ l^ a b c d', block2=False)

    err = "Only implemented up to 3pdm."
    with pytest.raises(ValueError, match=err):
        _ = mps.rdm('i^ j^ k^ l^ a b c d', block2=True)
