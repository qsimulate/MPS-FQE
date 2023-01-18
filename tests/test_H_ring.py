import os
import numpy as np
import fqe
import numpy
import pytest
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from openfermion.chem import make_atomic_ring

from mps_fqe.wavefunction import MPSWavefunction


def get_H_ring_data(amount_H):
    filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"Hring_{amount_H}.hdf5")
    molecule = make_atomic_ring(amount_H, 1.0, "sto-3g", atom_type="H", charge=0, filename=filename)

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


@pytest.mark.parametrize("amount_H", list(range(2, 8)))
def test_H_ring_evolve(amount_H):
    molecule = get_H_ring_data(amount_H)

    nele = molecule.n_electrons
    sz = molecule.multiplicity - 1
    norbs = molecule.n_orbitals
    h1, h2 = molecule.get_integrals()

    fqe_wf = fqe.Wavefunction([[nele, sz, norbs]])
    fqe_wf.set_wfn(strategy="hartree-fock")
    fqe_wf.normalize()

    hamiltonian = fqe.get_restricted_hamiltonian((h1, numpy.einsum("ijlk", -0.5 * h2)), e_0=molecule.nuclear_repulsion)
    assert np.isclose(molecule.hf_energy, fqe_wf.expectationValue(hamiltonian))

    dt = 0.1
    steps = 10
    evolved = fqe_wf
    for i in range(steps):
        evolved = evolved.time_evolve(dt, hamiltonian)
    assert np.isclose(molecule.hf_energy, evolved.expectationValue(hamiltonian))

    MPO = Hamiltonian(
        FCIDUMP(
            pg="c1",
            n_sites=norbs,
            n_elec=nele,
            twos=sz,
            h1e=h1,
            g2e=numpy.einsum("iklj", h2),
            const_e=molecule.nuclear_repulsion,
        ),
        flat=True,
    ).build_qc_mpo()

    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wf)
    assert np.isclose(molecule.hf_energy, mps.expectationValue(MPO))

    mps_evolved = MPSWavefunction.from_fqe_wavefunction(evolved)
    assert np.isclose(molecule.hf_energy, mps_evolved.expectationValue(MPO))

    mps_evolved_2 = mps.time_evolve(dt * steps, MPO, bdim=100, steps=steps, method='rk4')
    mps_evolved_2 /= mps_evolved_2.norm()

    assert np.isclose(molecule.hf_energy, mps_evolved_2.expectationValue(MPO))
    assert np.isclose(np.abs(mps_evolved_2.conj() @ mps_evolved), 1.0)

    mps_evolved_3 = mps.time_evolve(dt * steps, MPO, bdim=100, steps=steps)
    mps_evolved_3 /= mps_evolved_3.norm()

    assert np.isclose(molecule.hf_energy, mps_evolved_3.expectationValue(MPO))
    assert np.isclose(np.abs(mps_evolved_3.conj() @ mps_evolved), 1.0)
