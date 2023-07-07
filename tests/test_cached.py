import os
import tempfile

import fqe
import numpy
from mps_fqe.wavefunction import MPSWavefunction
from mps_fqe.hamiltonian import mpo_from_fqe_hamiltonian

from .test_H_ring import get_H_ring_data, hamiltonian_from_molecule


def test_propagation():
    mbd = 20
    amount_H = 7
    molecule = get_H_ring_data(amount_H)
    nele = molecule.n_electrons
    sz = molecule.multiplicity - 1
    norbs = molecule.n_orbitals

    fqe_wfn = fqe.Wavefunction([[nele, sz, norbs]])
    fqe_wfn.set_wfn(strategy="random")
    fqe_wfn.normalize()
    hamiltonian = hamiltonian_from_molecule(molecule)

    mpo = mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn, max_bond_dim=mbd)
    mps_evolved = mps.time_evolve(
        1, mpo, steps=10, method="tddmrg", cached=False)
    ref = mps_evolved.expectationValue(mpo)
    mps_cached = MPSWavefunction.from_fqe_wavefunction(fqe_wfn,
                                                       max_bond_dim=mbd)
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ['TMPDIR'] = str(temp_dir)
        mps_cached_evolved = mps_cached.time_evolve(
            1, mpo, steps=10, method="tddmrg", cached=True)
        out = mps_cached_evolved.expectationValue(mpo)

    assert numpy.isclose(ref, out)
