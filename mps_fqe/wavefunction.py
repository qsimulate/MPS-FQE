import os
import functools
import itertools
import tempfile
from typing import List, Union, Optional, Tuple

import numpy
import fqe
from fqe.hamiltonians.hamiltonian import Hamiltonian as FqeHamiltonian
from fqe.hamiltonians.sparse_hamiltonian import SparseHamiltonian
from openfermion import FermionOperator
from pyblock3.algebra.flat import FlatSparseTensor
from pyblock3.algebra.mps import MPS, MPSInfo
from pyblock3.algebra.mpe import MPE, CachedMPE
from pyblock3.algebra.integrate import rk4_apply
from pyblock3.algebra.symmetry import SZ
from pyblock3.block2.io import MPSTools
from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
from mps_fqe import utils
from mps_fqe.hamiltonian import mpo_from_fqe_hamiltonian


def decompose_mps(fci_tensor: FlatSparseTensor) -> List[FlatSparseTensor]:
    tensors = []

    while fci_tensor.ndim > 3:
        u, s, v = fci_tensor.tensor_svd(2, pattern="+"
                                        * (fci_tensor.ndim - 1) + "-")
        tensors.append(u)
        fci_tensor = numpy.tensordot(numpy.diag(s), v, axes=1)
    tensors.append(fci_tensor)

    return tensors


class MPSWavefunction(MPS):
    @classmethod
    def from_fqe_wavefunction(cls,
                              fqe_wfn: fqe.Wavefunction,
                              max_bond_dim: int = -1,
                              cutoff: float = 1E-12) \
            -> "MPSWavefunction":
        opts = {
            "max_bond_dim": max_bond_dim,
            "cutoff": cutoff
        }
        sectors = fqe_wfn.sectors()

        def get_fci_FlatSparseTensor(sectors):
            if not sectors:
                return None

            n_blocks = sum(fqe_wfn.get_coeff(sector).size
                           for sector in sectors)
            ndim = fqe_wfn.norb()
            physical_q_labels = list(map(SZ.to_flat,
                                         [SZ(0, 0, 0),
                                          SZ(1, 1, 0),
                                          SZ(1, -1, 0),
                                          SZ(2, 0, 0)]))

            shapes = numpy.ones((n_blocks, ndim + 2), dtype=numpy.uint32)

            q_labels = []
            data_array = []

            for sector in sectors:
                alpha_str = fqe_wfn.sector(sector).get_fcigraph()\
                                                  .string_alpha_all()\
                                                  .astype(int)
                beta_str = fqe_wfn.sector(sector).get_fcigraph()\
                                                 .string_beta_all()\
                                                 .astype(int)
                alpha_occ = numpy.array(
                    [tuple(reversed(bin(1 << ndim | x)[3:]))
                     for x in alpha_str], dtype=numpy.uint32)
                beta_occ = numpy.array(
                    [tuple(reversed(bin(1 << ndim | x)[3:]))
                     for x in beta_str], dtype=numpy.uint32)

                q_labels_temp = alpha_occ[:, None, :]\
                    + 2 * beta_occ[None, :, :]
                q_label = numpy.array(
                    [physical_q_labels[i] for i in
                     q_labels_temp.flatten()]).reshape(-1, ndim)

                fermionic_sign = utils.fqe_sign_change(fqe_wfn.sector(sector))
                data_array.append((fermionic_sign
                                   * fqe_wfn.get_coeff(sector)).flatten())

                vacuum = numpy.ones(len(q_label)) * SZ(0, 0, 0).to_flat()
                target = numpy.ones(len(q_label)) * SZ(sector[0],
                                                       sector[1],
                                                       0).to_flat()
                q_labels.append(numpy.hstack((vacuum[:, None],
                                              q_label,
                                              target[:, None])))

            q_labels_arr = numpy.vstack(q_labels)
            data = numpy.concatenate(data_array)

            return FlatSparseTensor(q_labels_arr, shapes, data)

        fci_tensor = get_fci_FlatSparseTensor(sectors)
        tensors = decompose_mps(fci_tensor)

        return cls(tensors=tensors, opts=opts)

    def to_fqe_wavefunction(self,
                            broken: Optional[Union[List[str], str]] = None)\
            -> fqe.Wavefunction:
        """This is quite a memory intensive function.
        Will fail for all but the smallest MPS's."""
        fci_tensor = self.get_FCITensor()
        norb = fci_tensor.ndim - 2
        sectors = tuple(map(SZ.from_flat, set(fci_tensor.q_labels[:, -1])))

        param = tuple((sector.n, sector.twos, norb) for sector in sectors)
        fqe_wfn = fqe.Wavefunction(param, broken)

        for qlabel, shape, data in zip(fci_tensor.q_labels,
                                       fci_tensor.shapes,
                                       fci_tensor.data):
            assert numpy.prod(shape) == 1
            qlabel = tuple(SZ.from_flat(x) for x in qlabel)
            sector = fqe_wfn.sector((qlabel[-1].n, qlabel[-1].twos))
            astring = sum(2 ** n if x in (SZ(1, 1, 0), SZ(2, 0, 0))
                          else 0 for n, x in enumerate(qlabel[1:-1]))
            bstring = sum(2 ** n if x in (SZ(1, -1, 0), SZ(2, 0, 0))
                          else 0 for n, x in enumerate(qlabel[1:-1]))
            aindx = sector.get_fcigraph().index_alpha(astring)
            bindx = sector.get_fcigraph().index_beta(bstring)
            sector.coeff[aindx, bindx] = data

        for sector in fqe_wfn.sectors():
            coeff = fqe_wfn.get_coeff(sector)
            coeff *= utils.fqe_sign_change(fqe_wfn.sector(sector))

        return fqe_wfn

    @classmethod
    def from_pyblock3_mps(cls,
                          mps: MPS,
                          max_bond_dim: int = -1,
                          cutoff: float = 1E-12) -> "MPSWavefunction":
        opts = {
            'max_bond_dim': max_bond_dim,
            'cutoff': cutoff
        }

        return cls(tensors=mps.tensors, opts=opts,
                   dq=mps.dq, const=mps.const)

    def print_wfn(self) -> None:
        for ii, tensor in enumerate(self.tensors):
            print(f"TENSOR {ii}")
            print(tensor)

        print("BOND DIMENSIONS")
        print(self.show_bond_dims())

    def norb(self) -> int:
        return self.n_sites

    def canonicalize(self, center) -> "MPSWavefunction":
        return self.from_pyblock3_mps(super().canonicalize(center))

    def compress(self, **opts) -> Tuple["MPSWavefunction", float]:
        mps, merror = super().compress(**opts)

        return self.from_pyblock3_mps(mps), merror

    def apply_linear(self,
                     mpo: MPS,
                     n_sweeps: int = 4,
                     cutoff: float = 0.0) -> "MPSWavefunction":
        bra = self.copy()
        mps = self.copy()
        bdim = self.opts['max_bond_dim']
        noises = [0.0]
        bdims = [bdim]

        MPE(bra, mpo - mpo.const, mps).linear(
            bdims=bdims, noises=noises,
            cg_thrds=None, iprint=0, n_sweeps=n_sweeps, tol=cutoff)
        bra += mpo.const*mps
        return type(self)(tensors=bra.tensors, opts=self.opts)

    def apply_exact(self, mpo: MPS) -> "MPSWavefunction":
        mps = self.copy()
        mps = mpo @ mps + 0*mps

        # It may still be possible to get an integer zero here.
        # For now, we raise a RuntimeError
        if isinstance(mps, int):
            assert mps == 0
            raise RuntimeError("Integer zero obtained when applying MPO")

        return type(self)(tensors=mps.tensors, opts=mps.opts)

    def apply(self,
              hamiltonian: Union[FqeHamiltonian, MPS],
              exact: bool = True) -> "MPSWavefunction":
        if isinstance(hamiltonian, FqeHamiltonian):
            hamiltonian = mpo_from_fqe_hamiltonian(hamiltonian,
                                                   n_sites=self.n_sites)
        if hamiltonian.n_sites != self.n_sites:
            raise ValueError('Hamiltonian has incorrect size:'
                             + ' expected {}'.format(self.n_sites)
                             + ' provided {}'.format(hamiltonian.n_sites))

        if exact:
            return self.apply_exact(hamiltonian)
        return self.apply_linear(hamiltonian)

    def transform(self):
        pass

    def tddmrg(self,
               time: float,
               hamiltonian: MPS,
               steps: int = 1,
               n_sub_sweeps: int = 1,
               cached: bool = False,
               cutoff: float = 1E-14) -> "MPSWavefunction":
        dt = time / steps
        mps = self.copy()

        mpe = CachedMPE(mps, hamiltonian, mps) if cached \
            else MPE(mps, hamiltonian, mps)
        bdim = mps.opts.get("max_bond_dim", -1)

        mpe.tddmrg(bdims=[bdim], dt=-dt * 1j, iprint=0, n_sweeps=steps,
                   normalize=False, n_sub_sweeps=n_sub_sweeps, cutoff=cutoff)

        mps += 0*self
        return type(self)(tensors=mps.tensors, opts=self.opts)

    def rk4_apply(self, time: float, hamiltonian: MPS,
                  steps: int = 1) -> "MPSWavefunction":
        dt = time / steps
        mps = self.copy()

        for ii in range(steps):
            mps = rk4_apply((-dt * 1j) * hamiltonian, mps)

        return type(self)(tensors=mps.tensors, opts=mps.opts)

    def rk4_apply_linear(self,
                         time: float,
                         hamiltonian: MPS,
                         steps: int = 1,
                         n_sub_sweeps: int = 1,
                         cutoff: float = 0.0) -> "MPSWavefunction":
        dt = -1.j * time / steps
        tmp = self.copy()
        mps = type(self)(tensors=tmp.tensors, opts=self.opts)
        for ii in range(steps):
            k1 = dt * mps.apply_linear(
                hamiltonian, n_sweeps=n_sub_sweeps, cutoff=cutoff)

            k2 = 0.5 * k1 + mps
            k2 = type(self)(tensors=k2.tensors, opts=mps.opts)
            k2 = dt * k2.apply_linear(
                hamiltonian, n_sweeps=n_sub_sweeps, cutoff=cutoff)

            k3 = 0.5 * k2 + mps
            k3 = type(self)(tensors=k3.tensors, opts=mps.opts)
            k3 = dt * k3.apply_linear(
                hamiltonian, n_sweeps=n_sub_sweeps, cutoff=cutoff)

            k4 = k3 + mps
            k4 = type(self)(tensors=k4.tensors, opts=mps.opts)
            k4 = dt * k4.apply_linear(
                hamiltonian, n_sweeps=n_sub_sweeps, cutoff=cutoff)

            mps = mps + (k1 + 2*k2 + 2*k3 + k4)/6
            mps = type(self)(tensors=mps.tensors, opts=mps.opts)

        return mps

    def time_evolve(self,
                    time: float,
                    hamiltonian: Union[FqeHamiltonian, MPS],
                    inplace: bool = False,
                    steps: int = 1,
                    n_sub_sweeps: int = 1,
                    method: str = "tddmrg",
                    cached: bool = False,
                    cutoff: float = 1E-14) -> "MPSWavefunction":
        if isinstance(hamiltonian, FqeHamiltonian):
            hamiltonian = mpo_from_fqe_hamiltonian(hamiltonian,
                                                   n_sites=self.n_sites)
        if method.lower() == "tddmrg":
            return self.tddmrg(time, hamiltonian, steps,
                               n_sub_sweeps=n_sub_sweeps,
                               cached=cached, cutoff=cutoff)
        if method.lower() == "rk4":
            return self.rk4_apply(time, hamiltonian, steps)
        if method.lower() == "rk4-linear":
            return self.rk4_apply_linear(time, hamiltonian, steps=steps,
                                         n_sub_sweeps=n_sub_sweeps,
                                         cutoff=cutoff)
        raise ValueError(
            f"method needs to be 'tddmrg' or 'rk4', '{method}' given")

    def expectationValue(self, hamiltonian: MPS,
                         brawfn: Optional["MPSWavefunction"] = None) -> float:
        bra = self if brawfn is None else brawfn
        if isinstance(hamiltonian, FqeHamiltonian):
            hamiltonian = mpo_from_fqe_hamiltonian(hamiltonian,
                                                   n_sites=self.n_sites)
        return MPE(bra, hamiltonian, self)[0:2].expectation

    def get_FCITensor(self) -> FlatSparseTensor:
        return functools.reduce(lambda x, y: numpy.tensordot(x, y, axes=1),
                                self.tensors)

    def scale(self, sval: complex) -> None:
        """ Scale each configuration space by the value sval

        Args:
            sval (complex): value to scale by
        """
        self.tensors[0] = sval*self.tensors[0]

    def rdm(self, string: str, brawfn: Optional["MPSWavefunction"] = None,
            block2: bool = True) -> Union[complex, numpy.ndarray]:
        # Get an individual rdm element
        if any(char.isdigit() for char in string):
            mpo = mpo_from_fqe_hamiltonian(
                SparseHamiltonian(FermionOperator(string)),
                n_sites=self.n_sites)
            return self.expectationValue(mpo, brawfn)

        rank = len(string.split()) // 2
        if len(string.split()) % 2 != 0:
            raise ValueError("RDM must have even number of operators.")
        if block2:
            if brawfn is not None:
                raise ValueError("Transition density not implemented \
                with block2 driver.")
            return self._block2_rdm(rank)
        # Get the entire rdm
        if rank == 1:
            return self._get_rdm1(brawfn)
        if rank == 2:
            return self._get_rdm2(brawfn)
        if rank == 3:
            return self._get_rdm3(brawfn)
        raise ValueError("RDM is only implemented up to 3pdm.")

    def _get_rdm1(self, brawfn):
        rdm1 = numpy.zeros((self.n_sites, self.n_sites), dtype=complex)
        for isite in range(self.n_sites):
            for jsite in range(isite+1):
                mpo = utils.one_body_projection_mpo(isite, jsite,
                                                    self.n_sites)
                rdm1[isite, jsite] = self.expectationValue(mpo,
                                                           brawfn=brawfn)
        return rdm1 + numpy.tril(rdm1, -1).transpose().conjugate()

    def _get_rdm2(self, brawfn):
        rdm2 = numpy.zeros((self.n_sites, self.n_sites,
                            self.n_sites, self.n_sites), dtype=complex)
        for isite, jsite, ksite, lsite in itertools.product(
                range(self.n_sites), repeat=4):
            mpo = utils.two_body_projection_mpo(isite, jsite,
                                                ksite, lsite,
                                                self.n_sites)
            rdm2[isite, jsite, ksite, lsite] = \
                self.expectationValue(mpo, brawfn)
        return rdm2

    def _get_rdm3(self, brawfn):
        rdm3 = numpy.zeros((self.n_sites, self.n_sites,
                            self.n_sites, self.n_sites,
                            self.n_sites, self.n_sites), dtype=complex)
        for isite, jsite, ksite, lsite, msite, nsite in itertools.product(
                range(self.n_sites), repeat=6):
            mpo = utils.three_body_projection_mpo(isite, jsite,
                                                  ksite, lsite,
                                                  msite, nsite,
                                                  self.n_sites)
            rdm3[isite, jsite, ksite, lsite, msite, nsite] = \
                self.expectationValue(mpo, brawfn)
        return rdm3

    def _block2_rdm(self, rank):
        if rank > 3:
            raise ValueError("Only implemented up to 3pdm.")
        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ['TMPDIR'] = str(temp_dir)
            driver = DMRGDriver(scratch=os.environ['TMPDIR'],
                                symm_type=SymmetryTypes.SZ | SymmetryTypes.CPX,
                                n_threads=1)
            driver.initialize_system(n_sites=self.n_sites,
                                     orb_sym=[0]*self.n_sites)
            b2mps = MPSTools.to_block2(self, save_dir=driver.scratch)
            if rank == 1:
                rdm = numpy.sum(driver.get_1pdm(b2mps), axis=0)
            elif rank == 2:
                b2rdm = driver.get_2pdm(b2mps)
                rdm = b2rdm[0] + b2rdm[2] \
                    - (numpy.einsum("ijlk", b2rdm[1])
                       + numpy.einsum("jikl", b2rdm[1]))
            elif rank == 3:
                b2rdm = driver.get_3pdm(b2mps)
                rdm = b2rdm[0] + b2rdm[3] \
                    + numpy.einsum("ijklmn->ijkmnl", b2rdm[1]) \
                    + numpy.einsum("ijklmn->ikjmln", b2rdm[1]) \
                    + numpy.einsum("ijklmn->kijlmn", b2rdm[1]) \
                    + numpy.einsum("ijklmn->ijknlm", b2rdm[2]) \
                    + numpy.einsum("ijklmn->jiklnm", b2rdm[2]) \
                    + numpy.einsum("ijklmn->jkilmn", b2rdm[2])
        return rdm


def get_hf_mps(nele, sz, norbs, bdim,
               e0: float = 0.0,
               cutoff: float = 0.0,
               full: bool = True,
               occ: Optional[list[int]] = None) -> "MPSWavefunction":
    if (nele + abs(sz)) // 2 > norbs:
        raise ValueError(
            f"Electron number is too large (nele = {nele}, norb = {norbs})")
    if sz % 2 != nele % 2:
        raise ValueError(
            f"Spin (sz = {sz}) is incompatible with nele = {nele}")

    fd = FCIDUMP(pg='c1',
                 n_sites=norbs,
                 const_e=e0,
                 n_elec=nele,
                 twos=sz)
    nsocc = abs(sz)
    ndocc = (nele - nsocc) // 2
    nvirt = norbs - nsocc - ndocc
    assert nvirt >= 0
    if occ is None:
        occ = [2]*ndocc + [1]*nsocc + [0]*nvirt
    occd = occ.count(2)
    occs = occ.count(1)
    occv = occ.count(0)
    if occd != ndocc:
        raise ValueError(
            "Inconsistent doubly occupied orbitals: {occd} ({docc})")
    if occs != nsocc:
        raise ValueError(
            "Inconsistent singly occupied orbitals: {occs} ({socc})")
    if occv != nvirt:
        raise ValueError(
            "Inconsistent virtual orbitals: {occv} ({virt})")

    hamil = Hamiltonian(fd, flat=True)
    mps_info = MPSInfo(hamil.n_sites, hamil.vacuum, hamil.target, hamil.basis)
    mps_info.set_bond_dimension_occ(bdim, occ=occ)
    mps_wfn = MPS.ones(mps_info)
    if full:
        mps_info_full = MPSInfo(
            hamil.n_sites, hamil.vacuum, hamil.target, hamil.basis)
        mps_info_full.set_bond_dimension(bdim)
        mps_wfn += 0*MPS.ones(mps_info_full)
    return MPSWavefunction.from_pyblock3_mps(mps_wfn, max_bond_dim=bdim,
                                             cutoff=cutoff)
