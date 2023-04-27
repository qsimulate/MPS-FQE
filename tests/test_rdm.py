import numpy
from numpy import einsum

import pytest
import fqe
from openfermion import FermionOperator
from mps_fqe.wavefunction import MPSWavefunction
from mps_fqe.hamiltonian import mpo_from_fqe_hamiltonian


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

    # assert numpy.isclose(mps.rdm('1^ 5'),
    #                      fqe_wfn.rdm('1^ 5'),
    #                      atol=1e-12)


@pytest.mark.parametrize("n_electrons,sz,n_orbitals", [(2, 0, 2),
                                                       (6, 0, 4)])
def test_rdm2(n_electrons, sz, n_orbitals):
    fqe_wfn = fqe.Wavefunction([[n_electrons, sz, n_orbitals]])
    fqe_wfn.set_wfn(strategy='random')
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn=fqe_wfn)
    assert numpy.allclose(mps.rdm('i^ j^ k l'),
                          fqe_wfn.rdm('i^ j^ k l'),
                          atol=1E-12)
    # assert numpy.isclose(mps.rdm("0^ 2^ 0 2"),
    #                      fqe_wfn.rdm("0^ 2^ 0 2"),
    #                      atol=1E-12)
