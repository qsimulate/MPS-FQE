import io
import sys
from itertools import product
from copy import deepcopy

import scipy as sp
import numpy as np
from pyscf import ao2mo, scf, gto
import openfermion as of
from pyblock3.algebra.mps import MPS
from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian
from fqe.hamiltonians.sparse_hamiltonian import SparseHamiltonian
from mps_fqe.wavefunction import get_hf_mps, MPSWavefunction
from mps_fqe.hamiltonian import mpo_from_fqe_hamiltonian


def get_N2_parameters(r, basis='sto6g'):
    geom = "{}\t{}\t{}\t{}\n".format("N", 0, 0, 0)
    geom += "{}\t{}\t{}\t{}\n".format("N", r, 0, 0)
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    nele = mol.nelectron
    sz = mol.spin
    e_0 = mol.energy_nuc()
    norbs = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norbs)
    return nele, sz, e_0, h1, h2


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
        self.op_pool: list[of.FermionOperator] = []

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
                 rk4_bound: int = 0,
                 verbose: bool = True):
        self.hamiltonian = hamiltonian
        self.force_norm_thresh = force_norm_thresh
        self.max_iteration = max_iteration
        self.rk4_bound = rk4_bound
        self.selected_operator_indices = []
        self.selected_operator_coefficients = []
        self.force_vec_norms = []
        self.energies = []
        self.pool = op_pool.op_pool
        self.verbose = verbose

    def run(self, initial_wfn: MPSWavefunction, exact_apply: bool = True):
        iteration = 0
        rk4 = "rk4" if exact_apply else "rk4-linear"
        if self.verbose:
            with open("ADAPT.log", 'a') as fout:
                fout.write("Starting ADAPT-MPS calculation with: \n")
                fout.write(f"force norm thresh = {self.force_norm_thresh}\n")
                fout.write(f"max iteration = {self.max_iteration}\n")
                fout.write(f"rk4 bound = {self.rk4_bound}\n")

        while iteration < self.max_iteration:
            # Get the initial state
            wfn = deepcopy(initial_wfn)

            # Evolve with the selected set of operators with current coeffs
            for i, coeff in enumerate(self.selected_operator_coefficients):
                method = "tddmrg" if i >= self.rk4_bound else rk4
                op_index = self.selected_operator_indices[i]
                mpo, _ = mpo_from_fqe_hamiltonian(
                    self.pool[op_index], n_sites=wfn.n_sites).compress(
                        cutoff=wfn.opts["cutoff"])
                wfn = wfn.time_evolve(time=1j*coeff,
                                      hamiltonian=mpo,
                                      method=method,
                                      steps=1,
                                      add_noise=True)
            self.energies.append(wfn.expectationValue(self.hamiltonian).real)

            # Calculate all of the gradients
            gradients = self.compute_pool_gradients(wfn, exact_apply)
            self.force_vec_norms.append(np.linalg.norm(gradients).real)
            if self.verbose:
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
            if iteration > 0:
                self.selected_operator_coefficients \
                    = self.optimize_coefficients(initial_wfn, exact_apply)
            iteration += 1

        if self.verbose:
            with open("ADAPT.log", 'a') as fout:
                fout.write("Reached max iteration before convergence\n")

        return iteration, self.energies[-1]

    def optimize_coefficients(self,
                              initial_wfn: MPSWavefunction,
                              exact_apply: bool = True):
        rk4 = "rk4" if exact_apply else "rk4-linear"

        def cost_function(coeffs):
            phi = deepcopy(initial_wfn)
            gradients = np.zeros_like(coeffs)
            for i, coeff in enumerate(coeffs):
                method = "tddmrg" if i >= self.rk4_bound else rk4
                op_index = self.selected_operator_indices[i]
                mpo, _ = mpo_from_fqe_hamiltonian(
                    self.pool[op_index], n_sites=phi.n_sites).compress(
                        cutoff=phi.opts["cutoff"])
                phi = phi.time_evolve(time=1j*coeff,
                                      hamiltonian=mpo,
                                      method=method,
                                      steps=1,
                                      add_noise=True)
            energy = phi.expectationValue(self.hamiltonian).real
            sigma = phi.apply(self.hamiltonian, exact=exact_apply)
            sigma, _ = sigma.compress(max_bond_dim=sigma.opts["max_bond_dim"],
                                      cutoff=sigma.opts["cutoff"])

            for i in range(-1, -(len(gradients)+1), -1):
                method = "tddmrg" if abs(i) >= self.rk4_bound else rk4
                op_index = self.selected_operator_indices[i]
                mpo, _ = mpo_from_fqe_hamiltonian(
                    self.pool[op_index], n_sites=phi.n_sites).compress(
                        cutoff=phi.opts["cutoff"])
                gradients[i] = 2.0 * phi.expectationValue(mpo,
                                                          brawfn=sigma).real
                coeff = coeffs[i]
                phi = phi.time_evolve(time=-1j*coeff,
                                      hamiltonian=mpo,
                                      method=method,
                                      steps=1,
                                      add_noise=True)
                sigma = sigma.time_evolve(time=-1j*coeff,
                                          hamiltonian=mpo,
                                          method=method,
                                          steps=1,
                                          add_noise=True)
            return (energy, np.array(gradients, order="F"))

        res = sp.optimize.minimize(cost_function,
                                   self.selected_operator_coefficients,
                                   method="L-BFGS-B",
                                   jac=True)
        return [x for x in res.x]

    def compute_pool_gradients(self, wfn, exact_apply):
        #divert output
        save_stdout = sys.stdout
        sys.stdout = io.StringIO()

        grads = np.zeros_like(self.pool)
        H_wfn = wfn.apply(self.hamiltonian, exact=exact_apply)
        for i, op in enumerate(self.pool):
            mpo, _ = mpo_from_fqe_hamiltonian(
                op, n_sites=wfn.n_sites).compress(cutoff=wfn.opts["cutoff"])
            grads[i] = wfn.expectationValue(mpo, brawfn=H_wfn) \
                - H_wfn.expectationValue(mpo, brawfn=wfn)

        sys.stdout = save_stdout
        return grads



if __name__ == '__main__':
    # ADAPT-MPS parameters
    rk4_bound = 10
    exact_apply = True

    # Molecular parameters
    basis = "sto6g"
    rs = [0.9 + i*0.1 for i in range(11)]
    nele, sz, _, h1, _ = get_N2_parameters(rs[0], basis)
    norbs = len(h1[0])
    assert abs(sz) == 0
    nocc = nele//2
    occ = list(range(nocc))
    virt = list(range(nocc, norbs))

    # Get OperatorPool
    op_pool = OperatorPool(norbs, occ, virt)
    op_pool.one_body_sz_adapted()
    op_pool.two_body_sz_adapted()

    # MPSWavefunction parameters
    bdims = [100, 200, 300]
    cutoff = 1E-14
    for bdim in bdims:
        initial_wfn = get_hf_mps(nele, sz, norbs,
                                 bdim=bdim,
                                 cutoff=cutoff,
                                 dtype=complex)
        assert initial_wfn.opts["cutoff"] == cutoff
        assert initial_wfn.opts["max_bond_dim"] == bdim
        niters = []
        energies = []
        for r in rs:
            _, _, e_0, h1, h2 = get_N2_parameters(r, basis)
            fd = FCIDUMP(pg='c1', n_sites=norbs, n_elec=nele, twos=sz, ipg=0,
                         uhf=False, h1e=h1, g2e=h2, const_e=e_0)
            ham, _ = Hamiltonian(fd, flat=True).build_qc_mpo().compress(
                cutoff=cutoff)
            adapt = ADAPT(ham, op_pool, rk4_bound=rk4_bound)
            niter, E_final = adapt.run(initial_wfn, exact_apply=exact_apply)
            niters.append(niter)
            energies.append(E_final)
        with open(f"adapt_n2_{r}.out", 'w') as fout:
            for i, energy in enumerate(energies):
                fout.write(f"{rs[i]}\t{energy}\t{niters[i]}\n")
