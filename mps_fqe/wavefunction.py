import functools
import numpy as np
from typing import List, Union, Optional, Tuple

import fqe
from pyblock3.algebra.flat import FlatSparseTensor
from pyblock3.algebra.mps import MPS
from pyblock3.algebra.mpe import MPE
from pyblock3.algebra.integrate import rk4_apply
from pyblock3.algebra.symmetry import SZ


def decompose_mps(fci_tensor: FlatSparseTensor) -> List[FlatSparseTensor]:
    tensors = []

    while fci_tensor.ndim > 3:
        u, s, v = fci_tensor.tensor_svd(2, pattern="+" * (fci_tensor.ndim - 1) + "-")
        tensors.append(u)
        fci_tensor = np.tensordot(np.diag(s), v, axes=1)
    tensors.append(fci_tensor)

    return tensors


class MPSWavefunction(MPS):
    @classmethod
    def from_fqe_wavefunction(cls, fqe_wfn: fqe.Wavefunction) -> "MPSWavefunction":
        sectors = fqe_wfn.sectors()

        def get_fci_FlatSparseTensor(sectors):
            if not sectors:
                return None

            n_blocks = sum(fqe_wfn.get_coeff(sector).size for sector in sectors)
            ndim = fqe_wfn.norb()
            physical_q_labels = list(map(SZ.to_flat, [SZ(0, 0, 0), SZ(1, 1, 0), SZ(1, -1, 0), SZ(2, 0, 0)]))

            shapes = np.ones((n_blocks, ndim + 2), dtype=np.uint32)

            q_labels = []
            data_array = []
            for sector in sectors:
                alpha_str = fqe_wfn.sector(sector).get_fcigraph().string_alpha_all().astype(int)
                alpha_occ = np.array([tuple(reversed(bin(1 << ndim | x)[3:])) for x in alpha_str], dtype=np.uint32)
                beta_str = fqe_wfn.sector(sector).get_fcigraph().string_beta_all().astype(int)
                beta_occ = np.array([tuple(reversed(bin(1 << ndim | x)[3:])) for x in beta_str], dtype=np.uint32)
                q_labels_temp = alpha_occ[:, None, :] + 2 * beta_occ[None, :, :]
                q_label = np.array([physical_q_labels[i] for i in q_labels_temp.flatten()]).reshape(-1, ndim)

                cumsum_alpha_occ = np.cumsum(alpha_occ, axis=1)
                n_alpha, n_beta = sum(alpha_occ[0]), sum(beta_occ[0])
                swaps = np.sum(cumsum_alpha_occ[:, None, :] * beta_occ[None, :, :], axis=2) + (n_alpha * n_beta)
                fermionic_sign = np.where(swaps % 2 == 0, 1.0, -1.0)
                data_array.append((fermionic_sign * fqe_wfn.get_coeff(sector)).flatten())

                vacuum = np.ones(len(q_label)) * SZ(0, 0, 0).to_flat()
                target = np.ones(len(q_label)) * SZ(sector[0], sector[1], 0).to_flat()
                q_labels.append(np.hstack((vacuum[:, None], q_label, target[:, None])))

            q_labels_arr = np.vstack(q_labels)
            data = np.concatenate(data_array)
            return FlatSparseTensor(q_labels_arr, shapes, data)

        fci_tensor = get_fci_FlatSparseTensor(sectors)
        return cls(tensors=decompose_mps(fci_tensor))

    def to_fqe_wavefunction(self, broken: Optional[Union[List[str], str]] = None) -> fqe.Wavefunction:
        """This is quite a memory intensive function. Will fail for all but the smallest MPS's."""
        fci_tensor = self.get_FCITensor()
        norb = fci_tensor.ndim - 2
        sectors = tuple(map(SZ.from_flat, set(fci_tensor.q_labels[:, -1])))

        param = tuple((sector.n, sector.twos, norb) for sector in sectors)
        fqe_wfn = fqe.Wavefunction(param, broken)

        for qlabel, shape, data in zip(fci_tensor.q_labels, fci_tensor.shapes, fci_tensor.data):
            assert np.prod(shape) == 1
            qlabel = tuple(SZ.from_flat(x) for x in qlabel)
            sector = fqe_wfn.sector((qlabel[-1].n, qlabel[-1].twos))
            astring = sum(2 ** n if x in (SZ(1, 1, 0), SZ(2, 0, 0)) else 0 for n, x in enumerate(qlabel[1:-1]))
            bstring = sum(2 ** n if x in (SZ(1, -1, 0), SZ(2, 0, 0)) else 0 for n, x in enumerate(qlabel[1:-1]))
            aindx = sector.get_fcigraph().index_alpha(astring)
            bindx = sector.get_fcigraph().index_beta(bstring)
            sector.coeff[aindx, bindx] = data

        return fqe_wfn

    @classmethod
    def from_pyblock3_mps(cls, mps: MPS) -> "MPSWavefunction":
        return cls(tensors=mps.tensors)

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

        return self.from_pyblock3_mps(mps), merror

    def apply(self):
        pass

    def transform(self):
        pass

    def tddmrg(self, time: float, hamiltonian: MPS, bdim: int = 100, steps: Optional[int] = None):
        steps = 1 if steps is None else steps
        dt = time / steps
        mps = self.copy()
        mpe = MPE(mps, hamiltonian, mps)
        mpe.tddmrg(bdims=[bdim], dt=-dt * 1j, iprint=0, n_sweeps=steps, normalize=False)
        return type(self)(tensors=mps.tensors)

    def rk4_apply(
        self, time: float, hamiltonian: MPS, bdim: int = 100, steps: Optional[int] = None
    ) -> "MPSWavefunction":
        steps = 1 if steps is None else steps
        dt = time / steps
        mps = self.copy()
        mps.opts = dict(max_bond_dim=bdim)
        for ii in range(steps):
            mps = rk4_apply((-dt * 1j) * hamiltonian, mps)
        return type(self)(tensors=mps.tensors)

    def time_evolve(
        self, time: float, hamiltonian: MPS, bdim: int = 100, steps: Optional[int] = None, method: str = "tddmrg"
    ) -> "MPSWavefunction":
        if method.lower() == "tddmrg":
            return self.tddmrg(time, hamiltonian, bdim, steps)
        elif method.lower() == "rk4":
            return self.rk4_apply(time, hamiltonian, bdim, steps)
        else:
            raise ValueError(f"method needs to be 'tddmrg' or 'rk4', '{method}' given")

    def expectationValue(self, hamiltonian: MPS) -> float:
        return MPE(self, hamiltonian, self)[0:2].expectation

    def get_FCITensor(self) -> FlatSparseTensor:
        return functools.reduce(lambda x, y: np.tensordot(x, y, axes=1), self.tensors)
