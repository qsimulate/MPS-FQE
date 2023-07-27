import itertools
import os
import fqe
import numpy as np
from openfermion.chem import make_atomic_ring
import pytest

from mps_fqe.wavefunction import MPSWavefunction, get_hf_mps
from mps_fqe.hamiltonian import mpo_from_fqe_hamiltonian


def get_H_ring_data(amount_H):
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            f"Hring_{amount_H}.hdf5")
    molecule = make_atomic_ring(amount_H, 1.0, "sto-3g",
                                atom_type="H", charge=0, filename=filename)

    if os.path.isfile(filename):
        molecule.load()

    if molecule.hf_energy is None:
        molecule = generate_H_ring_data(molecule)
    return molecule


def generate_H_ring_data(molecule):
    from openfermionpyscf import run_pyscf

    molecule = run_pyscf(molecule, run_scf=True)
    molecule.save()
    return molecule


def hamiltonian_from_molecule(molecule):
    h1, h2 = molecule.get_integrals()
    e_0 = molecule.nuclear_repulsion
    hamiltonian = fqe.get_restricted_hamiltonian((h1,
                                                  np.einsum("ijlk",
                                                            -0.5 * h2)),
                                                 e_0=e_0)
    return hamiltonian


@pytest.mark.parametrize("amount_H,method",
                         itertools.product(range(2, 7),
                                           ["rk4-linear", "rk4", "tddmrg"]))
def test_H_ring_evolve(amount_H, method):
    molecule = get_H_ring_data(amount_H)

    nele = molecule.n_electrons
    sz = molecule.multiplicity - 1
    norbs = molecule.n_orbitals
    hamiltonian = hamiltonian_from_molecule(molecule)

    fqe_wf = fqe.Wavefunction([[nele, sz, norbs]])
    fqe_wf.set_wfn(strategy="hartree-fock")
    fqe_wf.normalize()

    assert np.isclose(molecule.hf_energy, fqe_wf.expectationValue(hamiltonian))

    dt = 0.05
    steps = 20
    total_time = dt*steps
    tddmrg_steps = 10
    sub_sweeps = 2
    rk4_steps = 30
    bdim = 4 ** ((amount_H + 1) // 2)

    evolved = fqe_wf
    for _ in range(steps):
        evolved = evolved.time_evolve(dt, hamiltonian)
    assert np.isclose(molecule.hf_energy,
                      evolved.expectationValue(hamiltonian))

    mpo = mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)
    mps = get_hf_mps(nele, sz, norbs, bdim=bdim)
    assert np.isclose(molecule.hf_energy, mps.expectationValue(mpo))

    mps_evolved = MPSWavefunction.from_fqe_wavefunction(evolved)
    assert np.isclose(molecule.hf_energy, mps_evolved.expectationValue(mpo))

    mps_evolved_to_fqe = mps_evolved.to_fqe_wavefunction()
    for sector in evolved.sectors():
        assert np.allclose(evolved.get_coeff(sector),
                           mps_evolved_to_fqe.get_coeff(sector))

    if method == 'tddmrg':
        mps_evolved_2 = mps.time_evolve(
            total_time, mpo, steps=tddmrg_steps,
            method=method, n_sub_sweeps=sub_sweeps)
    elif method == 'rk4':
        mps_evolved_2 = mps.time_evolve(
            total_time, mpo, steps=rk4_steps, method=method)
    else:
        assert method == 'rk4-linear'
        mps_evolved_2 = mps.time_evolve(
            total_time, mpo, steps=rk4_steps,
            n_sub_sweeps=sub_sweeps, method=method)

    assert np.isclose(molecule.hf_energy, mps_evolved_2.expectationValue(mpo))
    global_phase_shift = mps_evolved_2.conj() @ mps_evolved
    assert np.isclose(np.abs(global_phase_shift), 1.0)

    mps_evolved_2_to_fqe = mps_evolved_2.to_fqe_wavefunction()
    for sector in evolved.sectors():
        ref = evolved.get_coeff(sector)
        out = global_phase_shift*mps_evolved_2_to_fqe.get_coeff(sector)
        assert np.allclose(ref, out, atol=1e-06)
