import os
import numpy
from numpy import einsum

import fqe
from mps_fqe.wavefunction import MPSWavefunction
from mps_fqe.hamiltonian import mpo_from_fqe_hamiltonian
from test_H_ring import get_H_ring_data


os.environ['TMPDIR'] = './tmpdir'

def test_propagation():
    mbd = 20
    amount_H = 7
    molecule = get_H_ring_data(amount_H)
    nele = molecule.n_electrons
    sz = molecule.multiplicity - 1
    norbs = molecule.n_orbitals
    h1, h2 = molecule.get_integrals()

    fqe_wfn = fqe.Wavefunction([[nele, sz, norbs]])
    fqe_wfn.set_wfn(strategy="random")
    fqe_wfn.normalize()
    e_0 = molecule.nuclear_repulsion
    hamiltonian = fqe.get_restricted_hamiltonian((h1,
                                                  einsum("ijlk", -0.5 * h2)),
                                                 e_0=e_0)

    mpo = mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn, max_bond_dim=mbd)
    mps_cached = MPSWavefunction.from_fqe_wavefunction(fqe_wfn,
                                                       max_bond_dim=mbd)
    mps_evolved = mps.time_evolve(1, mpo, 10, "tddmrg", cached=False)
    mps_cached_evolved = mps_cached.time_evolve(1, mpo, 10,
                                                "tddmrg", cached=True)
    assert numpy.isclose(mps_evolved.expectationValue(mpo),
                         mps_cached_evolved.expectationValue(mpo))
