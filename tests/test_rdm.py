import os
import numpy
import tempfile

import pytest
import fqe
fqe.settings.use_accelerated_code = False
from mps_fqe.wavefunction import MPSWavefunction


@pytest.mark.parametrize("n_electrons,sz,n_orbitals", [(2, 2, 4),
                                                       (6, -2, 4),
                                                       (5, 1, 4),
                                                       (6, 2, 4),
                                                       (4, 0, 6)])
def test_rdm1(n_electrons, sz, n_orbitals):
    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    fqe_wfn.set_wfn(strategy='random')
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)
    # Spatial orbitals
    assert numpy.allclose(mps.rdm('i^ j'),
                          fqe_wfn.rdm('i^ j'),
                          atol=1e-12)
    # Spin orbitals
    assert numpy.isclose(mps.rdm('0^ 2'),
                         fqe_wfn.rdm('0^ 2'),
                         atol=1e-12)

    assert numpy.isclose(mps.rdm('1^ 3'),
                         fqe_wfn.rdm('1^ 3'),
                         atol=1e-12)


@pytest.mark.parametrize("n_electrons,sz,n_orbitals", [(2, 0, 2),
                                                       (6, 0, 4)])
def test_rdm2(n_electrons, sz, n_orbitals):
    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    fqe_wfn.set_wfn(strategy='random')
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)
    mps_rdm = mps.rdm('i^ j^ k l')
    fqe_rdm = fqe_wfn.rdm('i^ j^ k l')

    assert numpy.isclose(mps.rdm('0^ 1^ 0 3'),
                         fqe_wfn.rdm('0^ 1^ 0 3'),
                         atol=1e-12)

    assert numpy.isclose(mps.rdm('0^ 1^ 2 1'),
                         fqe_wfn.rdm('0^ 1^ 2 1'),
                         atol=1e-12)

    assert numpy.allclose(mps.rdm('i^ j^ k l'),
                          fqe_wfn.rdm('i^ j^ k l'),
                          atol=1E-12)


@pytest.mark.parametrize("n_electrons,sz,n_orbitals", [(2, 2, 4),
                                                       (6, -2, 4),
                                                       (5, 1, 4),
                                                       (6, 2, 4),
                                                       (4, 0, 6)])
def test_block2_rdm1(n_electrons, sz, n_orbitals):
    from pyblock3.block2.io import MPSTools
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    fqe_wfn.set_wfn(strategy='random')
    fqe_wfn.normalize()
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)

    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['TMPDIR'] = str(temp_dir)
        driver = DMRGDriver(scratch=os.environ['TMPDIR'],
                            symm_type=SymmetryTypes.SZ | SymmetryTypes.CPX,
                            n_threads=1)
        driver.initialize_system(n_sites=mps.n_sites,
                                 orb_sym=[0]*mps.n_sites)
        b2mps = MPSTools.to_block2(mps, save_dir=driver.scratch)
        rdm1 = numpy.sum(driver.get_1pdm(b2mps), axis=0)

    fqe_rdm1 = fqe_wfn.rdm("i^ j")

    assert numpy.allclose(rdm1,
                          fqe_rdm1,
                          atol=1E-12)


# @pytest.mark.parametrize("n_electrons,sz,n_orbitals", [(2, 2, 4),
#                                                        (6, -2, 4),
#                                                        (5, 1, 4),
#                                                        (6, 2, 4),
#                                                        (4, 0, 6)])
# def test_block2_rdm2(n_electrons, sz, n_orbitals):
#     from pyblock3.block2.io import MPSTools
#     from pyblock2.driver.core import DMRGDriver, SymmetryTypes

#     fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
#     fqe_wfn.set_wfn(strategy='random')
#     fqe_wfn.normalize()
#     mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)

#     with tempfile.TemporaryDirectory() as temp_dir:
#         os.environ['TMPDIR'] = str(temp_dir)
#         driver = DMRGDriver(scratch=os.environ['TMPDIR'],
#                             symm_type=SymmetryTypes.SZ | SymmetryTypes.CPX,
#                             n_threads=1)
#         driver.initialize_system(n_sites=mps.n_sites,
#                                  orb_sym=[0]*mps.n_sites)
#         b2mps = MPSTools.to_block2(mps, save_dir=driver.scratch)
#         rdm2 = sum(driver.get_2pdm(b2mps))

#     fqe_rdm2 = fqe_wfn.rdm("i^ j^ k l")

#     print(rdm2[0][1][0][0])
#     print(fqe_rdm2[0][0][0][1])
#     # assert numpy.allclose(rdm2,
#     #                       fqe_rdm2,
#     #                       atol=1E-12)
