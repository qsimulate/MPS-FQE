import numpy
import pytest
import fqe
from mps_fqe.wavefunction import MPSWavefunction, get_random_mps
from mps_fqe.hamiltonian import mpo_from_fqe_hamiltonian
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


@pytest.mark.parametrize("nele,sz,norb", [(6, -2, 4),
                                          (6, 2, 4),
                                          (2, 2, 4),
                                          (5, 1, 4)])
def test_gen_rdm1(nele, sz, norb):
    numpy.random.seed(11)
    mps = get_random_mps(nele, sz, norb, bdim=50)
    h1 = numpy.random.random((norb, norb))
    h1 += h1.transpose()
    hamiltonian = fqe.get_restricted_hamiltonian((h1,),
                                                 e_0=0)

    mpo = mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)
    ref = mps.expectationValue(mpo)
    rdm1 = mps.rdm('i^ j', block2=False)
    out = numpy.einsum('ij,ij', rdm1, h1)
    assert abs(ref - out) < 1e-12


@pytest.mark.parametrize("nele,sz,norb", [(6, -2, 4),
                                          (6, 2, 4),
                                          (2, 2, 4),
                                          (5, 1, 4)])
def test_gen_rdm2(nele, sz, norb):
    numpy.random.seed(22)
    mps = get_random_mps(nele, sz, norb, bdim=50)

    # eight-fold symmetry for real 2-electron operator
    h2 = numpy.random.random((norb, norb, norb, norb))
    h2 += h2.transpose((1, 0, 3, 2))
    h2 += h2.transpose((0, 1, 3, 2))
    h2 += h2.transpose((1, 0, 2, 3))
    h2 += h2.transpose((2, 3, 0, 1))
    h2 += h2.transpose((3, 2, 1, 0))
    h2 += h2.transpose((2, 3, 1, 0))
    h2 += h2.transpose((3, 2, 0, 1))

    hamiltonian = fqe.get_restricted_hamiltonian(
        (numpy.zeros((norb, norb)), h2), e_0=0)

    mpo = mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)
    ref = mps.expectationValue(mpo)
    rdm1 = mps.rdm('i^ j^ k l', block2=False)
    out = numpy.einsum('ijkl,ijkl', rdm1, h2)
    assert abs(ref - out) < 1e-12


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
