import copy
from itertools import product
import random

import pytest
import numpy as np
from fqe.wavefunction import Wavefunction
from fqe.hamiltonians.sparse_hamiltonian import SparseHamiltonian
from fqe.util import vdot
from .test_H_ring import get_H_ring_data, \
    hamiltonian_from_molecule
from mps_fqe.wavefunction import MPSWavefunction, get_hf_mps, \
    get_random_mps
from mps_fqe.hamiltonian import mpo_from_fqe_hamiltonian
from openfermion import FermionOperator
from openfermion.utils import hermitian_conjugated


def get_fqe_operators(mat: np.ndarray, hermitian: bool = True):
    sign = 1 if hermitian else -1
    norbs = mat.shape[0]
    fqe_ops = []
    for i, j in product(range(norbs), repeat=2):
        if j <= i:
            continue
        if np.isclose(mat[i, j], 0):
            continue
        of_op = FermionOperator(((2*i, 1), (2*j, 0)),
                                coefficient=mat[i, j])
        of_op += FermionOperator(((2*i+1, 1), (2*j+1, 0)),
                                 coefficient=mat[i, j])
        of_op += sign * hermitian_conjugated(of_op)
        fqe_ops.append(SparseHamiltonian(of_op))
    return fqe_ops


@pytest.mark.parametrize("with_mpo", [True, False])
def test_H_ring_evolve(with_mpo):
    amount_H = 6
    molecule = get_H_ring_data(amount_H)
    nele = molecule.n_electrons
    norbs = molecule.n_orbitals
    sz = molecule.multiplicity - 1
    bdim = 4 ** ((amount_H + 1) // 2)
    hamiltonian = hamiltonian_from_molecule(molecule)
    op = mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)\
        if with_mpo else hamiltonian
    dt = 0.05
    tddmrg_steps = 10
    sub_sweeps = 2
    total_time = dt*tddmrg_steps*sub_sweeps

    init_mps = get_hf_mps(nele, sz, norbs, bdim=bdim)
    block2_evolved = init_mps.time_evolve(total_time, op,
                                          steps=tddmrg_steps,
                                          n_sub_sweeps=sub_sweeps,
                                          block2=True)
    pyblock_evolved = init_mps.time_evolve(total_time, op,
                                           steps=tddmrg_steps,
                                           n_sub_sweeps=sub_sweeps,
                                           block2=False)
    phase = block2_evolved.conj() @ pyblock_evolved
    assert np.isclose(phase, 1.0)


@pytest.mark.parametrize("time_axis,strategy",
                         product(["real", "imaginary"],
                                 ["random", "hartree-fock"]))
def test_sparse_operator_evolve(time_axis, strategy):
    t = 0.1 if time_axis == "real" else 0.1j
    norbs = 4
    nele = 4
    sz = 0
    steps = 1
    n_sub_sweeps = 1
    max_bond_dim = 4 ** (norbs+1 // 2)
    add_noise = strategy == "hartree-fock"
    hermitian = time_axis == "real"

    k1_triu = np.triu_indices(norbs, k=1)
    nvars = norbs * (norbs-1) // 2
    random_variables = [random.random() for _ in range(nvars)]
    mat = np.zeros((norbs, norbs))
    mat[k1_triu] = random_variables
    mat += mat.T

    fqe_wfn = Wavefunction([[nele, sz, norbs]])
    fqe_wfn.set_wfn(strategy=strategy)

    mps_wfn = MPSWavefunction.from_fqe_wavefunction(fqe_wfn)
    fqe_ops = get_fqe_operators(mat, hermitian)

    for fqe_op in fqe_ops:
        fqe_evolved = fqe_wfn.time_evolve(t, fqe_op)
        fqe_ovlp = vdot(fqe_evolved, fqe_wfn)

        mpo = mpo_from_fqe_hamiltonian(fqe_op, norbs)
        block2_evolved = mps_wfn._block2_tddmrg(time=t, hamiltonian=mpo,
                                                steps=steps,
                                                n_sub_sweeps=n_sub_sweeps,
                                                cutoff=0, iprint=0,
                                                add_noise=add_noise)
        block2_ovlp = block2_evolved.conj() @ mps_wfn

        pyblock_evolved = mps_wfn.tddmrg(time=t, hamiltonian=mpo, steps=steps,
                                         n_sub_sweeps=n_sub_sweeps, cutoff=0,
                                         block2=False)
        pyblock_ovlp = pyblock_evolved.conj() @ mps_wfn

        assert np.isclose(fqe_ovlp, block2_ovlp)
        assert np.isclose(block2_ovlp, pyblock_ovlp)


@pytest.mark.parametrize("time_axis",
                         ["real", "imaginary"])
def test_add_noise(time_axis):
    t = 0.1 if time_axis == "real" else 0.1j
    norbs = 4
    nele = 4
    sz = 0
    steps = 1
    n_sub_sweeps = 1
    max_bond_dim = 4 ** (norbs+1 // 2)
    hermitian = time_axis == "real"

    k1_triu = np.triu_indices(norbs, k=1)
    nvars = norbs * (norbs-1) // 2
    random_variables = [random.random() for _ in range(nvars)]
    mat = np.zeros((norbs, norbs))
    mat[k1_triu] = random_variables
    mat += mat.T

    mps_wfn = get_random_mps(nele, sz, norbs, max_bond_dim)
    fqe_ops = get_fqe_operators(mat, hermitian)

    evolved = copy.deepcopy(mps_wfn)
    evolved_noise = copy.deepcopy(mps_wfn)
    for fqe_op in fqe_ops:
        mpo = mpo_from_fqe_hamiltonian(fqe_op, norbs)
        evolved = evolved._block2_tddmrg(time=t, hamiltonian=mpo,
                                         steps=steps,
                                         n_sub_sweeps=n_sub_sweeps,
                                         cutoff=0, iprint=0,
                                         add_noise=False)
        evolved_noise = evolved_noise._block2_tddmrg(time=t, hamiltonian=mpo,
                                                     steps=steps,
                                                     n_sub_sweeps=n_sub_sweeps,
                                                     cutoff=0, iprint=0,
                                                     add_noise=True)
        phase = evolved.conj() @ evolved_noise
        assert np.isclose(phase, evolved.norm()**2)
