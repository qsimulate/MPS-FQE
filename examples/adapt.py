import io
import sys
from itertools import product
from copy import deepcopy

import scipy as sp
import numpy as np
from pyscf import ao2mo, scf, gto, fci
import openfermion as of
from pyblock3.algebra.mps import MPS
from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian
from fqe.hamiltonians.sparse_hamiltonian import SparseHamiltonian
from mps_fqe.wavefunction import MPSWavefunction, get_hf_mps


def get_H2_parameters(r, basis='sto3g'):
    geom = "{}\t{}\t{}\t{}\n".format("H", 0, 0, 0)
    geom += "{}\t{}\t{}\t{}\n".format("H", r, 0, 0)
    geom += "{}\t{}\t{}\t{}\n".format("H", 2*r, 0, 0)
    geom += "{}\t{}\t{}\t{}\n".format("H", 3*r, 0, 0)
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    nele = mol.nelectron
    e_0 = mol.energy_nuc()
    norbs = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norbs)
    fci_energy = fci.FCI(mf).kernel()[0]
    return (h1, h2), e_0, nele, mf.e_tot, fci_energy


class OperatorPool:
    def __init__(self, norbs: int, occ: list[int], virt: list[int]):
        """
        Routines for defining operator pools

        Args:
            norbs: number of spatial orbitals
            occ: list of indices of the occupied orbitals
            virt: list of indices of the virtual orbitals
        """
        self.norbs = norbs
        self.occ = occ
        self.virt = virt
        self.op_pool: list[SparseHamiltonian] = []

    def two_body_sz_adapted(self):
        """
        Doubles generators each with distinct Sz expectation value.

        G^{isigma, jtau, ktau, lsigma) for sigma, tau in 0, 1
        """
        for i, j, k, l in product(range(self.norbs), repeat=4):
            if i < j and k < l:
                op_aa = ((2 * i, 1), (2 * j, 1), (2 * k, 0), (2 * l, 0))
                op_bb = ((2 * i + 1, 1), (2 * j + 1, 1), (2 * k + 1, 0),
                         (2 * l + 1, 0))
                fop_aa = of.FermionOperator(op_aa)
                fop_aa = fop_aa - of.hermitian_conjugated(fop_aa)
                fop_bb = of.FermionOperator(op_bb)
                fop_bb = fop_bb - of.hermitian_conjugated(fop_bb)
                fop_aa = of.normal_ordered(fop_aa)
                fop_bb = of.normal_ordered(fop_bb)
                self.op_pool.append(SparseHamiltonian(fop_aa))
                self.op_pool.append(SparseHamiltonian(fop_bb))

            op_ab = ((2 * i, 1), (2 * j + 1, 1), (2 * k + 1, 0), (2 * l, 0))
            fop_ab = of.FermionOperator(op_ab)
            fop_ab = fop_ab - of.hermitian_conjugated(fop_ab)
            fop_ab = of.normal_ordered(fop_ab)
            if not np.isclose(fop_ab.induced_norm(), 0):
                self.op_pool.append(SparseHamiltonian(fop_ab))

    def one_body_sz_adapted(self):
        """alpha-alpha and beta-beta rotations
        """
        for i, j in product(range(self.norbs), repeat=2):
            if i > j:
                op_aa = ((2 * i, 1), (2 * j, 0))
                op_bb = ((2 * i + 1, 1), (2 * j + 1, 0))
                fop_aa = of.FermionOperator(op_aa)
                fop_aa = fop_aa - of.hermitian_conjugated(fop_aa)
                fop_bb = of.FermionOperator(op_bb)
                fop_bb = fop_bb - of.hermitian_conjugated(fop_bb)
                fop_aa = of.normal_ordered(fop_aa)
                fop_bb = of.normal_ordered(fop_bb)
                self.op_pool.append(SparseHamiltonian(fop_aa))
                self.op_pool.append(SparseHamiltonian(fop_bb))


class ADAPT():
    def __init__(self, hamiltonian: MPS,
                 op_pool: OperatorPool,
                 force_norm_thresh: float = 1E-3,
                 max_iteration: int = 100,
                 prop_steps: int = 8,
                 verbose: bool = True):
        self.hamiltonian = hamiltonian
        self.force_norm_thresh = force_norm_thresh
        self.max_iteration = max_iteration
        self.selected_operator_indices = []
        self.selected_operator_coefficients = []
        self.force_vec_norms = []
        self.energies = []
        self.pool = op_pool.op_pool
        self.verbose = verbose
        self.steps = prop_steps

    def run(self, initial_wfn: MPSWavefunction):
        iteration = 0
        if self.verbose:
            with open("ADAPT.log", 'a') as fout:
                fout.write("Starting ADAPT calculation with: \n")
                fout.write(f"force norm thresh = {self.force_norm_thresh}\n")
                fout.write(f"max iteration = {self.max_iteration}\n")
                fout.write("max bond dimension = "
                           f"{initial_wfn.opts['max_bond_dim']}\n")

        while iteration < self.max_iteration:
            # Get the initial state
            wfn = deepcopy(initial_wfn)
            # Evolve with the selected set of operators with current coeffs
            for i, coeff in enumerate(self.selected_operator_coefficients):
                op = self.pool[self.selected_operator_indices[i]]
                for _ in range(self.steps):
                    wfn = wfn.time_evolve(time=1j*coeff/self.steps,
                                          hamiltonian=op,
                                          method="rk4")
                    wfn, _ = wfn.compress(
                        max_bond_dim=initial_wfn.opts["max_bond_dim"],
                        cutoff=initial_wfn.opts["cutoff"])

            self.energies.append(wfn.expectationValue(self.hamiltonian).real)

            # Calculate all of the gradients
            gradients = self.compute_pool_gradients(wfn)
            self.force_vec_norms.append(np.linalg.norm(gradients).real)

            with open("ADAPT.log", 'a') as fout:
                fout.write(f"{iteration}\t"
                           f"{self.energies[iteration]}\t"
                           f"{self.force_vec_norms[iteration]}\n")

            # If converged, end simulation
            if self.force_vec_norms[-1] < self.force_norm_thresh:
                return iteration, self.energies[-1]

            # Identify the largest magnitude
            maxval_index = np.argmax(np.abs(gradients))
            self.selected_operator_indices.append(maxval_index)
            self.selected_operator_coefficients.append(0.0)

            # Perform the optimization
            self.selected_operator_coefficients \
                = self.optimize_coefficients(initial_wfn)
            iteration += 1

        if self.verbose:
            with open("ADAPT.log", 'a') as fout:
                fout.write("Reached max iteration before convergence\n")

    def optimize_coefficients(self, initial_wfn: MPSWavefunction):
        # Define cost function
        def cost_function(coeffs):
            phi = deepcopy(initial_wfn)
            gradients = np.zeros_like(coeffs)
            for i, coeff in enumerate(coeffs):
                op = self.pool[self.selected_operator_indices[i]]
                for _ in range(self.steps):
                    phi = phi.time_evolve(time=1j*coeff/self.steps,
                                          hamiltonian=op,
                                          method="rk4")
                    phi, _ = phi.compress(
                        max_bond_dim=initial_wfn.opts["max_bond_dim"],
                        cutoff=initial_wfn.opts["cutoff"])

            energy = phi.expectationValue(self.hamiltonian).real
            sigma = phi.apply_linear(self.hamiltonian, n_sweeps=8)
            for i in range(-1, -(len(gradients)+1), -1):
                op = self.pool[self.selected_operator_indices[i]]
                gradients[i] = 2.0 * phi.expectationValue(op,
                                                          brawfn=sigma).real
                for _ in range(self.steps):
                    phi = phi.time_evolve(time=-1j*coeffs[i]/self.steps,
                                          hamiltonian=op,
                                          method="rk4")
                    phi, _ = phi.compress(
                        max_bond_dim=initial_wfn.opts["max_bond_dim"],
                        cutoff=initial_wfn.opts["cutoff"])
                    sigma = sigma.time_evolve(time=-1j*coeffs[i]/self.steps,
                                              hamiltonian=op,
                                              method="rk4")
                    sigma, _ = sigma.compress(
                        max_bond_dim=initial_wfn.opts["max_bond_dim"],
                        cutoff=initial_wfn.opts["cutoff"])
            return (energy, np.array(gradients, order="F"))

        res = sp.optimize.minimize(cost_function,
                                   self.selected_operator_coefficients,
                                   method="BFGS",
                                   jac=True)
        return [x for x in res.x]

    def compute_pool_gradients(self, wfn):
        # Divert output
        save_stdout = sys.stdout
        sys.stdout = io.StringIO()
        grads = np.zeros_like(self.pool)
        # Get H|\psi>
        H_wfn = wfn.apply_linear(self.hamiltonian, n_sweeps=8)
        # Compute commutators
        for i, op in enumerate(self.pool):
            grads[i] = 2*wfn.expectationValue(op, brawfn=H_wfn)

        sys.stdout = save_stdout
        return grads.real


if __name__ == '__main__':
    # Molecular parameters
    basis = "sto3g"
    rs = [0.9, 1.1, 1.3, 1.5, 1.7]
    (h1, h2), e_0, nele, _, _ = get_H2_parameters(rs[0], basis)
    norbs = len(h1[0])
    nocc = nele//2
    occ = list(range(nocc))
    virt = list(range(nocc, norbs))
    sz = 0

    # Get OperatorPool
    op_pool = OperatorPool(norbs, occ, virt)
    op_pool.one_body_sz_adapted()
    op_pool.two_body_sz_adapted()

    # MPSWavefunction parameters
    bdims = [20, 50]
    cutoff = 1e-20
    # Do calculation for each of the bond dims
    for bdim in bdims:
        initial_wfn = get_hf_mps(nele, sz, norbs,
                                 bdim=bdim,
                                 cutoff=cutoff,
                                 dtype=complex)
        # Get potential
        for r in rs:
            (h1, h2), e_0, nele, E_hf, E_fci = get_H2_parameters(r, basis)
            print(E_hf, E_fci)
            fd = FCIDUMP(pg='c1', n_sites=norbs, n_elec=nele, twos=sz, ipg=0,
                         uhf=False, h1e=h1, g2e=h2, const_e=e_0)
            ham, _ = Hamiltonian(fd, flat=True).build_qc_mpo().compress(
                cutoff=cutoff)
            adapt = ADAPT(ham, op_pool, prop_steps=4)
            adapt.run(initial_wfn)
            with open(f"MPS_{bdim}_{round(r,1)}.out", 'w') as fout:
                fout.write("#iteration energy force_norm\n")
                fout.write("#Ehf: {}\tEfci: {}\n".format(E_hf, E_fci))
                for i, (e, f) in enumerate(zip(adapt.energies,
                                               adapt.force_vec_norms)):
                    fout.write("{}\t{}\t{}\n".format(i, e, f))
