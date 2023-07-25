import fqe
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
    hamiltonian = hamiltonian_from_molecule(molecule)

    fqe_wfn = fqe.Wavefunction([[nele, sz, norbs]])
    fqe_wfn.set_wfn(strategy="random")
    fqe_wfn.normalize()

    mpo = mpo_from_fqe_hamiltonian(fqe_ham=hamiltonian)
    mps = MPSWavefunction.from_fqe_wavefunction(fqe_wfn, max_bond_dim=mbd)
    # before_propagation
    assert mps.bond_dim == 64
    mps_evolved = mps.time_evolve(10, mpo, steps=10, method="tddmrg")
    assert mps_evolved.bond_dim == mbd
