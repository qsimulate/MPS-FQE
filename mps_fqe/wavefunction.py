import functools
import numpy as np
from typing import List, Union, Optional, Tuple

import fqe
from fqe.hamiltonians.hamiltonian import Hamiltonian as FqeHamiltonian
from pyblock3.algebra.flat import FlatSparseTensor
from pyblock3.algebra.mps import MPS
from pyblock3.algebra.mpe import MPE, CachedMPE
from pyblock3.algebra.integrate import rk4_apply
from pyblock3.algebra.symmetry import SZ

from mps_fqe import utils
from mps_fqe.hamiltonian import mpo_from_fqe_hamiltonian


def decompose_mps(fci_tensor: FlatSparseTensor) -> List[FlatSparseTensor]:
    tensors = []

    while fci_tensor.ndim > 3:
        u, s, v = fci_tensor.tensor_svd(2, pattern="+"
                                        * (fci_tensor.ndim - 1) + "-")
        tensors.append(u)
        fci_tensor = np.tensordot(np.diag(s), v, axes=1)
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
            'max_bond_dim': max_bond_dim,
            'cutoff': cutoff
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

            shapes = np.ones((n_blocks, ndim + 2), dtype=np.uint32)

            q_labels = []
            data_array = []

            for sector in sectors:
                alpha_str = fqe_wfn.sector(sector).get_fcigraph()\
                                                  .string_alpha_all()\
                                                  .astype(int)
                beta_str = fqe_wfn.sector(sector).get_fcigraph()\
                                                 .string_beta_all()\
                                                 .astype(int)
                alpha_occ = np.array([tuple(reversed(bin(1 << ndim | x)[3:]))
                                      for x in alpha_str], dtype=np.uint32)
                beta_occ = np.array([tuple(reversed(bin(1 << ndim | x)[3:]))
                                     for x in beta_str], dtype=np.uint32)

                q_labels_temp = alpha_occ[:, None, :]\
                    + 2 * beta_occ[None, :, :]
                q_label = np.array([physical_q_labels[i] for i in
                                    q_labels_temp.flatten()]).reshape(-1, ndim)

                fermionic_sign = utils.fqe_sign_change(fqe_wfn.sector(sector))
                data_array.append((fermionic_sign
                                   * fqe_wfn.get_coeff(sector)).flatten())

                vacuum = np.ones(len(q_label)) * SZ(0, 0, 0).to_flat()
                target = np.ones(len(q_label)) * SZ(sector[0],
                                                    sector[1],
                                                    0).to_flat()
                q_labels.append(np.hstack((vacuum[:, None],
                                           q_label,
                                           target[:, None])))

            q_labels_arr = np.vstack(q_labels)
            data = np.concatenate(data_array)

            return FlatSparseTensor(q_labels_arr, shapes, data)

        fci_tensor = get_fci_FlatSparseTensor(sectors)

        return cls(tensors=decompose_mps(fci_tensor), opts=opts)

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
            assert np.prod(shape) == 1
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

        return cls(tensors=mps.tensors, opts=opts)

    def print_wfn(self) -> None:
        for ii, tensor in enumerate(self.tensors):
            print(f"TENSOR {ii}")
            print(tensor)

        print("BOND DIMENSIONS")
        print(self.show_bond_dims())

    def canonicalize(self, center) -> "MPSWavefunction":
        return self.from_pyblock3_mps(super().canonicalize(center))

    def compress(self, **opts) -> Tuple["MPSWavefunction", float]:
        mps, merror = super().compress(**opts)
        max_bond_dim = opts.get('max_bond_dim', -1)
        cutoff = opts.get('cutoff', 1E-12)

        return self.from_pyblock3_mps(mps, max_bond_dim, cutoff), merror

    def apply(self,
              hamiltonian: Union[FqeHamiltonian, MPS]) -> "MPSWavefunction":
        if isinstance(hamiltonian, FqeHamiltonian):
            hamiltonian = mpo_from_fqe_hamiltonian(hamiltonian,
                                                   n_sites=self.n_sites)
        if hamiltonian.n_sites != self.n_sites:
            raise ValueError('Hamiltonian has incorrect size:'
                             + ' expected {}'.format(self.n_sites)
                             + ' provided {}'.format(hamiltonian.n_sites))
        mps = self.copy()
        mps = hamiltonian @ mps

        return type(self)(tensors=mps.tensors, opts=mps.opts)

    def transform(self):
        pass

    def tddmrg(self, time: float, hamiltonian: MPS,
               steps: int = 1, n_sub_sweeps: int = 1,
               cached: bool = False):
        dt = time / steps
        mps = self.copy()
        mpe = CachedMPE(mps, hamiltonian, mps) if cached \
            else MPE(mps, hamiltonian, mps)
        bdim = mps.opts.get("max_bond_dim", -1)

        try:
            mpe.tddmrg(bdims=[bdim], dt=-dt * 1j, iprint=0, n_sweeps=steps,
                       normalize=False, n_sub_sweeps=n_sub_sweeps)
        except RuntimeError as exc:
            pass

        return type(self)(tensors=mps.tensors, opts=mps.opts)

    def rk4_apply(self, time: float, hamiltonian: MPS,
                  steps: int = 1) -> "MPSWavefunction":
        dt = time / steps
        mps = self.copy()

        for ii in range(steps):
            mps = rk4_apply((-dt * 1j) * hamiltonian, mps)

        return type(self)(tensors=mps.tensors, opts=mps.opts)

    def time_evolve(self, time: float,
                    hamiltonian: Union[FqeHamiltonian, MPS],
                    inplace: bool = False,
                    steps: int = 1,
                    n_sub_sweeps: int = 1,
                    method: str = "tddmrg",
                    cached: bool = False) -> "MPSWavefunction":
        if isinstance(hamiltonian, FqeHamiltonian):
            hamiltonian = mpo_from_fqe_hamiltonian(hamiltonian,
                                                   n_sites=self.n_sites)
        if method.lower() == "tddmrg":
            return self.tddmrg(time, hamiltonian, steps, n_sub_sweeps=n_sub_sweeps, cached=cached)
        elif method.lower() == "rk4":
            return self.rk4_apply(time, hamiltonian, steps)
        else:
            raise ValueError(f"method needs to be 'tddmrg' or\
'rk4', '{method}' given")

    def expectationValue(self, hamiltonian: MPS) -> float:
        return MPE(self, hamiltonian, self)[0:2].expectation

    def get_FCITensor(self) -> FlatSparseTensor:
        return functools.reduce(lambda x, y: np.tensordot(x, y, axes=1),
                                self.tensors)
