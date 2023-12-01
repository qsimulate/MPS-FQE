import time
import pickle
import io
import os
import sys
from itertools import product
from copy import deepcopy

import scipy as sp
import numpy as np
from pyscf import ao2mo, scf, gto
import openfermion as of
from pyblock3.algebra.mps import MPS
from pyblock3.algebra.mpe import MPE
from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian
from fqe.hamiltonians.sparse_hamiltonian import SparseHamiltonian
from mps_fqe.wavefunction import get_hf_mps, get_random_mps,\
    MPSWavefunction
from mps_fqe.hamiltonian import mpo_from_fqe_hamiltonian


def get_N2_parameters(r, basis='sto3g'):
    geom = "{}\t{}\t{}\t{}\n".format("N", 0, 0, 0)
    geom += "{}\t{}\t{}\t{}\n".format("N", r, 0, 0)
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    nele = mol.nelectron
    e_0 = mol.energy_nuc()
    norbs = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), norbs)
    return (h1, h2), e_0, nele, mf.e_tot


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
                 verbose: bool = True,
                 finite_difference: bool = True):
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
        self.finite_difference = finite_difference

    def run(self, initial_wfn: MPSWavefunction, exact_apply: bool = True):
        iteration = 0
        mbd = initial_wfn.opts["max_bond_dim"]
        cutoff = initial_wfn.opts["cutoff"]
        print(mbd, cutoff)
        rk4 = "rk4" if exact_apply else "rk4-linear"
        if self.verbose:
            with open("ADAPT.log", 'a') as fout:
                fout.write("Starting ADAPT-MPS calculation with: \n")
                fout.write(f"max bond dimension = {initial_wfn.opts['max_bond_dim']}\n")
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
                mpo = mpo_from_fqe_hamiltonian(
                    self.pool[op_index], n_sites=wfn.n_sites,
                    add_identity=False)
                steps = 3*int(10*abs(coeff)) \
                    + 10*int(initial_wfn.apply(mpo).norm()) \
                    + 4
                print(f"propagating with {steps} steps")
                wfn = wfn.time_evolve(time=1j*coeff,
                                      hamiltonian=mpo,
                                      method=method,
                                      steps=steps,
                                      add_noise=True,
                                      block2=True)
                if method == "rk4":
                    wfn, _ = wfn.compress(
                        max_bond_dim=wfn.opts["max_bond_dim"],
                        cutoff=wfn.opts["cutoff"])
                    wfn.opts["max_bond_dim"] = mbd
                    wfn.opts["cutoff"] = cutoff

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
            mbd = initial_wfn.opts["max_bond_dim"]
            cutoff = initial_wfn.opts["cutoff"]
            print(mbd, cutoff)
            phi = deepcopy(initial_wfn)
            gradients = np.zeros_like(coeffs)
            for i, coeff in enumerate(coeffs):
                method = "tddmrg" if i >= self.rk4_bound else rk4
                op_index = self.selected_operator_indices[i]
                mpo = mpo_from_fqe_hamiltonian(
                    self.pool[op_index], n_sites=phi.n_sites,
                    add_identity=False)
                steps = 3*int(10*abs(coeff)) \
                    + 10*int(initial_wfn.apply(mpo).norm()) \
                    + 4
                print(f"propagating with {steps} steps")
                phi = phi.time_evolve(time=1j*coeff,
                                      hamiltonian=mpo,
                                      method=method,
                                      steps=steps,
                                      add_noise=True,
                                      block2=True)
                if method == "rk4":
                    phi, _ = phi.compress(
                        max_bond_dim=phi.opts["max_bond_dim"],
                        cutoff=phi.opts["cutoff"])
                    phi.opts["max_bond_dim"] = mbd
                    phi.opts["cutoff"] = cutoff
                
            assert np.isclose(phi.norm(), 1.0)
            energy = phi.expectationValue(self.hamiltonian).real
            if self.finite_difference:
                return energy

            sigma = phi.apply(self.hamiltonian, exact=False)
            for i in range(-1, -(len(gradients)+1), -1):
                method = "tddmrg" if abs(i) >= self.rk4_bound else rk4
                op_index = self.selected_operator_indices[i]
                mpo = mpo_from_fqe_hamiltonian(
                    self.pool[op_index], n_sites=phi.n_sites,
                    add_identity=False)
                steps = 3*int(10*abs(coeffs[i])) \
                    + 10*int(initial_wfn.apply(mpo).norm()) \
                    + 4
                print(f"propagating with {steps} steps")
                gradients[i] = 2.0 * phi.expectationValue(mpo,
                                                          brawfn=sigma).real
                mpo = mpo_from_fqe_hamiltonian(
                    self.pool[op_index], n_sites=phi.n_sites,
                    add_identity=False)
                phi = phi.time_evolve(time=-1j*coeffs[i],
                                      hamiltonian=mpo,
                                      method=method,
                                      steps=steps,
                                      add_noise=True,
                                      block2=True)
                sigma = sigma.time_evolve(time=-1j*coeffs[i],
                                          hamiltonian=mpo,
                                          method=method,
                                          steps=steps,
                                          add_noise=True,
                                          block2=True)
                # if method == "rk4":
                #     phi, _ = phi.compress(
                #         max_bond_dim=phi.opts["max_bond_dim"],
                #         cutoff=phi.opts["cutoff"])
                #     sigma, _ = sigma.compress(
                #         max_bond_dim=sigma.opts["max_bond_dim"],
                #         cutoff=sigma.opts["cutoff"])
            return (energy, np.array(gradients, order="F"))

        jac = self.finite_difference == False
        res = sp.optimize.minimize(cost_function,
                                   self.selected_operator_coefficients,
                                   method="BFGS",
                                   jac=jac)
        return [x for x in res.x]

    def compute_pool_gradients(self, wfn, exact_apply):
        # Divert output
        save_stdout = sys.stdout
        sys.stdout = io.StringIO()

        grads = np.zeros_like(self.pool)
        H_wfn = wfn.apply(self.hamiltonian, exact=False)
        for i, op in enumerate(self.pool):
            mpo = mpo_from_fqe_hamiltonian(
                op, n_sites=wfn.n_sites, add_identity=False)
            grads[i] = 2*wfn.expectationValue(mpo, brawfn=H_wfn)

        sys.stdout = save_stdout
        return grads.real

    def test_propagation(self, wfn, method):
        initial_wfn = deepcopy(wfn)
        print("Testing propagation with random pool elements")
        from random import randrange
        random_selections = [randrange(len(self.pool)) for _ in range(20)]
        coeffs = 0.01*np.random.rand(len(random_selections))
        for i, sel in enumerate(random_selections):
            mpo = mpo_from_fqe_hamiltonian(
                self.pool[sel], n_sites=wfn.n_sites, add_identity=False)
            tnot = time.perf_counter()
            steps = 3*int(10*abs(coeffs[i])) \
                + 10*int(initial_wfn.apply(mpo).norm()) \
                + 4
            print(f"propagating with {steps} steps")                            
            evolved1 = initial_wfn.time_evolve(time=coeffs[i]*1j,
                                               hamiltonian=mpo,
                                               method="rk4",
                                               steps=steps,
                                               block2=False)
            print(time.perf_counter() - tnot)

            tnot = time.perf_counter()
            steps = steps+20
            print(steps)
            evolved2 = initial_wfn.time_evolve(time=coeffs[i]*1j,
                                               hamiltonian=mpo,
                                               method="rk4",
                                               steps=steps)
            print(time.perf_counter() - tnot)

            assert np.isclose(evolved1.norm(), 1)
            assert np.isclose(evolved2.norm(), 1)
            ovlp1 = initial_wfn @ evolved1
            ovlp2 = initial_wfn @ evolved2
            print(ovlp1, ovlp2)
            assert np.isclose(ovlp1, ovlp2)
            init_wfn = evolved1
        print("Passed test propagation..")



if __name__ == '__main__':
    # ADAPT-MPS parameters
    rk4_bound = 100
    exact_apply = True

    # Molecular parameters
    basis = "6-31g"
    rs = [0.9 + i*0.2 for i in range(6)]
    (h1, h2), e_0, nele, _ = get_N2_parameters(rs[0], basis)
    sz = 0
    assert abs(sz) == 0
    norbs = len(h1[0])
    nocc = nele//2
    occ = list(range(nocc))
    virt = list(range(nocc, norbs))

    # Get OperatorPool
    op_pool = OperatorPool(norbs, occ, virt)
    op_pool.one_body_sz_adapted()
    op_pool.two_body_sz_adapted()

    # MPSWavefunction parameters
    bdims = [100]
    cutoff = 1e-20

    # Do calculation for each of the bond dims
    for bdim in bdims:
        initial_wfn = get_hf_mps(nele, sz, norbs,
                                 bdim=bdim,
                                 cutoff=cutoff,
                                 dtype=complex)
        assert initial_wfn.opts["max_bond_dim"] == bdim
        assert initial_wfn.opts["cutoff"] == cutoff
        # Get potential
        for r in rs:
            (h1, h2), e_0, nele, E_hf = get_N2_parameters(r, basis)
            fd = FCIDUMP(pg='c1', n_sites=norbs, n_elec=nele, twos=sz, ipg=0,
                         uhf=False, h1e=h1, g2e=h2, const_e=e_0)
            ham, _ = Hamiltonian(fd, flat=True).build_qc_mpo().compress(
                cutoff=cutoff)
            adapt = ADAPT(ham, op_pool, rk4_bound=rk4_bound)
            adapt.test_propagation(initial_wfn, method="tddmrg")
            niter, E_final = adapt.run(initial_wfn, exact_apply=exact_apply)
            with open(f"adapt_n2_{r}_{basis}.out", 'a') as fout:
                fout.write(f"{r}\t{E_final}\t{niter}\t{E_hf}\n")
                fout.flush()
