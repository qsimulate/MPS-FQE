import os
import numpy as np
import fqe
import numpy
import pytest
import itertools
from openfermion.chem import make_atomic_ring

from mps_fqe.wavefunction import MPSWavefunction
from mps_fqe.hamiltonian import MPOHamiltonian


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


@pytest.mark.parametrize("amount_H,method",
                         itertools.product(range(2, 9), ["rk4", "tddmrg"]))
def test_H_ring_evolve(amount_H, method):
    molecule = get_H_ring_data(amount_H)

    nele = molecule.n_electrons
    sz = molecule.multiplicity - 1
    norbs = molecule.n_orbitals
    h1, h2 = molecule.get_integrals()

    fqe_wf = fqe.Wavefunction([[nele, sz, norbs]])
    fqe_wf.set_wfn(strategy="hartree-fock")
    fqe_wf.normalize()

    e_0 = molecule.nuclear_repulsion
    hamiltonian = fqe.get_restricted_hamiltonian((h1,
                                                  numpy.einsum("ijlk",
                                                               -0.5 * h2)),
                                                 e_0=e_0)
    assert np.isclose(molecule.hf_energy, fqe_wf.expectationValue(hamiltonian))

    dt = 0.1
    steps = 10
    mini_dt = 0.0001
    mini_steps = 10
    bdim = 4 ** ((amount_H + 1) // 2)

    evolved = fqe_wf.time_evolve(mini_steps * mini_dt, hamiltonian)
    for i in range(steps):
        evolved = evolved.time_evolve(dt, hamiltonian)
    assert np.isclose(molecule.hf_energy,
                      evolved.expectationValue(hamiltonian))

    MPO = MPOHamiltonian.from_fqe_hamiltonian(fqe_ham=hamiltonian)
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wf).time_evolve(
        mini_dt * mini_steps, MPO, bdim=bdim, steps=mini_steps, method="rk4"
    )
    assert np.isclose(molecule.hf_energy, mps.expectationValue(MPO))

    mps_evolved = MPSWavefunction.from_fqe_wavefunction(evolved)
    assert np.isclose(molecule.hf_energy, mps_evolved.expectationValue(MPO))

    mps_evolved_2 = mps.time_evolve(dt * steps, MPO, bdim=bdim,
                                    steps=steps, method=method)
    mps_evolved_2 /= mps_evolved_2.norm()
    mps_evolved_2 = MPSWavefunction(tensors=mps_evolved_2.tensors)

    assert np.isclose(molecule.hf_energy, mps_evolved_2.expectationValue(MPO))
    global_phase_shift = mps_evolved_2.conj() @ mps_evolved
    assert np.isclose(np.abs(global_phase_shift), 1.0)

    mps_evolved_to_fqe = mps_evolved.to_fqe_wavefunction()
    for sector in evolved.sectors():
        assert np.allclose(evolved.get_coeff(sector),
                           mps_evolved_to_fqe.get_coeff(sector))

    mps_evolved_2_to_fqe = mps_evolved_2.to_fqe_wavefunction()
    for sector in evolved.sectors():
        assert np.allclose(
            evolved.get_coeff(sector),
            global_phase_shift * mps_evolved_2_to_fqe.get_coeff(sector),
            atol=1e-06
        )
