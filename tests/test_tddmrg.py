import itertools
import os
import fqe
import numpy as np
from openfermion.chem import make_atomic_ring
import pytest

from .test_H_ring import get_H_ring_data,\
    hamiltonian_from_molecule
from mps_fqe.wavefunction import MPSWavefunction, get_hf_mps
from mps_fqe.hamiltonian import mpo_from_fqe_hamiltonian


def test_H_ring_evolve():
    amount_H = 6
    molecule = get_H_ring_data(amount_H)
    nele = molecule.n_electrons
    norbs = molecule.n_orbitals
    sz = molecule.multiplicity - 1
    bdim = 4 ** ((amount_H + 1) // 2)
    hamiltonian = hamiltonian_from_molecule(molecule)
    mpo = mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)
    dt = 0.05
    tddmrg_steps = 10
    sub_sweeps = 2
    total_time = dt*tddmrg_steps*sub_sweeps

    init_mps = get_hf_mps(nele, sz, norbs, bdim=bdim)
    block2_evolved = init_mps.time_evolve(total_time, mpo,
                                          steps=tddmrg_steps,
                                          n_sub_sweeps=sub_sweeps,
                                          block2=True)
    pyblock_evolved = init_mps.time_evolve(total_time, mpo,
                                           steps=tddmrg_steps,
                                           n_sub_sweeps=sub_sweeps,
                                           block2=False)
    phase = block2_evolved.conj() @ pyblock_evolved
    assert np.isclose(phase, 1.0)
