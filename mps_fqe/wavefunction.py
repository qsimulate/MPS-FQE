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
from pyblock3.block2.io import MPSTools, MPOTools
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
    """MPSWavefunction is the central object for manipulation in \
    MPS-FQE. It is a 1-dimensional tensor network representation \
    of openfermion-FQE Wavefunction object to allow for approximate \
    description of large Fermionic systems.
    """

    @classmethod
    def from_fqe_wavefunction(cls,
                              fqe_wfn: fqe.Wavefunction,
                              max_bond_dim: Optional[int] = None,
                              cutoff: Optional[float] = None
                              ) -> "MPSWavefunction":
        """Construct an MPSWavefunction object from an openfermion-FQE \
        Wavefunction object. This classmethod iterates through the different \
        sectors of the Wavefunction object to obtain a list of \
        FlatSparseTensor objects for constructing the MPSWavefunction.

        Args:
            fqe_wfn (fqe.Wavefunction): an openfermion-FQE Wavefunction object.

            max_bond_dim (int): maximum bond dimension used to approximate \
                the MPS (optional, default -1).

            cutoff (float): threshold for discarded weights \
                (optional, default 1e-14).

        Returns:
            wfn (MPSWavefunction): an MPS representation of the \
                fqe.Wavefunction object.
        """
        # Get default pyblock options if not provided
        max_bond_dim = _default_pyblock_opts["max_bond_dim"] \
            if max_bond_dim is None else max_bond_dim
        cutoff = _default_pyblock_opts["cutoff"] \
            if cutoff is None else cutoff
        opts = {
            'max_bond_dim': max_bond_dim,
            'cutoff': cutoff
        }
        # Get default fqe options if not provided

        sectors = fqe_wfn.sectors()
        if not sectors:
            raise ValueError(
                "MPS cannot be constructed from empty Wavefunction")

        def get_fci_FlatSparseTensor(sectors):
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
        """Construct an fqe.Wavefunction object from the MPSWavefunction \
        object. This method is memory intensive, and will fail for anything \
        other than small systems.

        Args:
            broken (str): string of list of strings of broken symmetries.

        Returns:
            fqe_wfn (fqe.Wavefunction): an openfermion-FQE wavefunction.
        """
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
                          max_bond_dim: Optional[int] = None,
                          cutoff: Optional[float] = None) -> "MPSWavefunction":
        """Construct an MPSWavefunction object from a pyblock3 MPS object. \
        If optional arguments are not provided, the MPSWavefunction object \
        will inheret the necessary options from the pyblock3 MPS object.

        Args:
            mps (MPS): an openfermion-FQE Wavefunction object.

            max_bond_dim (int): maximum bond dimension used to approximate \
                the MPS (optional, default inhereted from MPS).

            cutoff (float): threshold for discarded weights \
                (optional, default inhereted from MPS).

        Returns:
            wfn (MPSWavefunction): an MPSWavefunction object.
        """
        # Use MPS object pyblock opts if not provided
        max_bond_dim = mps.opts["max_bond_dim"] if max_bond_dim is None \
            else max_bond_dim
        cutoff = mps.opts["cutoff"] if cutoff is None else cutoff
        opts = {
            'max_bond_dim': max_bond_dim,
            'cutoff': cutoff
        }

        return cls(tensors=mps.tensors, opts=opts)

    def print_wfn(self) -> None:
        """Print the current wavefunction tensors and the bond dimensions.
        """
        for ii, tensor in enumerate(self.tensors):
            print(f"TENSOR {ii}")
            print(tensor)

        print("BOND DIMENSIONS")
        print(self.show_bond_dims())

    def norb(self) -> int:
        """Return the number of spatial orbitals that underly the \
        wavefunction's state space.

        Returns:
            n_sites (int): number of spatial orbitals.
        """
        return self.n_sites

    def canonicalize(self, center) -> "MPSWavefunction":
        """
        """
        return self.from_pyblock3_mps(super().canonicalize(center))

    def compress(self, **opts) -> Tuple["MPSWavefunction", float]:
        """
        """
        mps, merror = super().compress(**opts)
        return self.from_pyblock3_mps(mps, opts["max_bond_dim"],
                                      opts["cutoff"]), merror

    def apply_linear(self,
                     mpo: MPS,
                     n_sweeps: int = 4,
                     cutoff: float = 0.0) -> "MPSWavefunction":
        """Apply an MPO to the MPSWavefunction object using an approximate \
        sweep algorithm.

        Args:
            mpo (MPS): a matrix product operator.

            n_sweeps (int): number of sweeps used to approximately apply the \
                mpo to the MPSWavefunction.

            cutoff (float): cutoff used during the sweep algorithm.

        Returns:
            wfn (MPSWavefunction): the resulting MPSWavefunction after the \
                mpo application.
        """
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
        """Apply an MPO to the MPSWavefunction object using exact \
        MPO-MPS contraction.

        Args:
            mpo (MPS): a matrix product operator.

        Returns:
            wfn (MPSWavefunction): the resulting MPSWavefunction after the \
                mpo application.
        """
        mps = self.copy()
        mps = mpo @ mps + 0*mps

        # This shouldn't happen now, so we check with an AssertionError
        if isinstance(mps, int):
            raise AssertionError

        return type(self)(tensors=mps.tensors, opts=self.opts)

    def apply(self,
              hamiltonian: Union[FqeHamiltonian, MPS],
              exact: bool = True) -> "MPSWavefunction":
        """Apply an MPO to the MPSWavefunction object.

        Args:
            hamiltonian (MPS or FqeHamiltonian): Operator to be applied.

            exact (bool): Whether to apply the operator exactly or via \
                an approximate sweep algorithm.

        Returns:
            wfn (MPSWavefunction): Resulting MPSWavefunction after \
                applying the operator.

        Raises:
            ValueError: If there is a mismatch between the number of sites \
                that define the MPSWavefunction and the MPO.
        """
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
        """Not currently implemented.
        """
        raise NotImplementedError

    def tddmrg(self,
               time: float,
               hamiltonian: MPS,
               **kwargs) -> "MPSWavefunction":
        """Perform real or imaginary time evolution under the influence of \
        the provided hamiltonian with TD-DMRG propagation. Details of the \
        propagation are controlled by keyword arguments. Relevant keyword \
        arguments and their defaults "steps": 1, "n_sub_sweeps": 1, \
        "cached": False, "block2": True, "add_noise": False, "cutoff": 1e-14

        Args:
            time (float): The total time of propagation.

            hamiltonian (MPS): MPO used to generate dynamics.

            \*\*kwargs (keyword arguments): Options to dictate propagation.

        Returns:
            wfn (MPSWavefunction): The time-evolved MPSWavefunction.
        """
        # Use provided options or else use object's assigned options
        block2 = kwargs.get("block2", _default_fqe_opts["block2"])
        steps = kwargs.get("steps", _default_fqe_opts["steps"])
        n_sub_sweeps = kwargs.get("n_sub_sweeps",
                                  _default_fqe_opts["n_sub_sweeps"])
        cached = kwargs.get("cached", _default_fqe_opts["cached"])
        cutoff = kwargs.get("cutoff", _default_pyblock_opts["cutoff"])
        if block2:
            add_noise = kwargs.get("add_noise", _default_fqe_opts["add_noise"])
            return self._block2_tddmrg(time=time, hamiltonian=hamiltonian,
                                       steps=steps, n_sub_sweeps=n_sub_sweeps,
                                       cutoff=cutoff, iprint=0,
                                       add_noise=add_noise)
        dt = time / steps
        mps = self.copy()

        mpe = CachedMPE(mps, hamiltonian, mps) if cached\
            else MPE(mps, hamiltonian, mps)
        bdim = mps.opts.get("max_bond_dim", -1)

        mpe.tddmrg(bdims=[bdim], dt=-dt * 1j, iprint=0, n_sweeps=steps,
                   normalize=False, n_sub_sweeps=n_sub_sweeps, cutoff=cutoff)

        mps += 0*self
        return type(self)(tensors=mps.tensors, opts=self.opts)

    def rk4_apply(self, time: float, hamiltonian: MPS,
                  **kwargs) -> "MPSWavefunction":
        """Perform real or imaginary time evolution under the influence of \
        the provided hamiltonian with RK4 propagation. Details of the \
        propagation are controlled by keyword arguments. Relevant keyword \
        arguments and their defaults "steps": 1.

        Args:
            time (float): The total time of propagation.

            hamiltonian (MPS): MPO used to generate dynamics.

            \*\*kwargs (keyword arguments): Options to dictate propagation.

        Returns:
            wfn (MPSWavefunction): The time-evolved MPSWavefunction.
        """
        # Use provided options or else use object's assigned options
        steps = kwargs.get("steps", _default_fqe_opts["steps"])
        dt = time / steps
        mps = self.copy()

        for ii in range(steps):
            mps = rk4_apply((-dt * 1j) * hamiltonian, mps)

        return type(self)(tensors=mps.tensors, const=self.const,
                          opts=self.opts)

    def rk4_apply_linear(self,
                         time: float,
                         hamiltonian: MPS,
                         **kwargs) -> "MPSWavefunction":
        """Perform real or imaginary time evolution under the influence of \
        the provided hamiltonian with a linear sweep algorithm approximation \
        to RK4 propagation. Details of the propagation are controlled \
        by keyword arguments. Relevant keyword arguments and their defaults \
        "steps": 1, "n_sub_sweeps": 1, "cutoff": 1e-14.

        Args:
            time (float): The total time of propagation.

            hamiltonian (MPS): MPO used to generate dynamics.

            \*\*kwargs (keyword arguments): Options to dictate propagation.

        Returns:
            wfn (MPSWavefunction): The time-evolved MPSWavefunction.
        """
        # Use provided options or else use object's assigned options
        steps = kwargs.get("steps", _default_fqe_opts["steps"])
        n_sub_sweeps = kwargs.get("n_sub_sweeps",
                                  _default_fqe_opts["n_sub_sweeps"])
        cutoff = kwargs.get("cutoff", _default_pyblock_opts["cutoff"])
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

        return type(self)(tensors=mps.tensors, opts=self.opts)

    def time_evolve(self,
                    time: float,
                    hamiltonian: Union[FqeHamiltonian, MPS],
                    inplace: bool = False,
                    **kwargs) -> "MPSWavefunction":
        """Perform real or imaginary time evolution under the influence of \
        the provided hamiltonian. Details of the propagation are controlled \
        by keyword arguments. Relevant keyword arguments and their defaults \
        "steps": 1, "n_sub_sweeps": 1, "method": "tddmrg", "cached": False, \
        "block2": True, "add_noise": False.

        Args:
            time (float): The total time of propagation.

            hamiltonian (MPS or FqeHamiltonian): Operator used to generate \
                dynamics.

            inplace (bool): Not implemented with the MPS backend.

            \*\*kwargs (keyword arguments): Options to dictate propagation.

        Returns:
            wfn (MPSWavefunction): The time-evolved MPSWavefunction.

        Raises:
            ValueError: If provided keyword method does not refer to an \
                implemented propagation method.
        """
        # Use provided options or else use default options
        method = kwargs.get("method", _default_fqe_opts["method"])
        options = kwargs.copy()

        if isinstance(hamiltonian, FqeHamiltonian):
            hamiltonian = mpo_from_fqe_hamiltonian(hamiltonian,
                                                   n_sites=self.n_sites)
        if method.lower() == "tddmrg":
            return self.tddmrg(time, hamiltonian, **options)
        if method.lower() == "rk4":
            return self.rk4_apply(time, hamiltonian, **options)
        if method.lower() == "rk4-linear":
            return self.rk4_apply_linear(time, hamiltonian, **options)
        raise ValueError(
            "method needs to be 'tddmrg', 'rk4', or 'rk4-linear',"
            f" '{method}' given")

    def expectationValue(self, hamiltonian: Union[MPS, FqeHamiltonian],
                         brawfn: Optional["MPSWavefunction"] = None) -> float:
        """Compute the expectation value of an operator.

        Args:
            hamiltonian (MPS or FqeHamiltonian): Operator whose expectation \
                value is measured.

            brawfn (MPSWavefunction): State used to compute off-diagonal \
                operator matrix elements (optional).

        Returns:
            expectation_value (complex): computed expectation value.
        """
        bra = self if brawfn is None else brawfn
        if isinstance(hamiltonian, FqeHamiltonian):
            hamiltonian = mpo_from_fqe_hamiltonian(hamiltonian,
                                                   n_sites=self.n_sites)
        return MPE(bra, hamiltonian, self)[0:2].expectation

    def get_FCITensor(self) -> FlatSparseTensor:
        """Get the full CI tensor description of the wave function.

        Returns:
            fci_tensor (FlatSparseTensor): Full CI tensor
        """
        return functools.reduce(lambda x, y: numpy.tensordot(x, y, axes=1),
                                self.tensors)

    def scale(self, sval: complex) -> None:
        """ Scale each configuration space by the value sval.

        Args:
            sval (complex): Value to scale by.
        """
        self.tensors[0] = sval*self.tensors[0]

    def rdm(self, string: str, brawfn: Optional["MPSWavefunction"] = None,
            block2: Optional[bool] = None) -> Union[complex, numpy.ndarray]:
        """ Returns the reduced density matrix (or elements) specified by the \
        string argument.

        Args:
            string (str): Character string specifying which rdm elements to \
                compute.

            brawfn (MPSWavefunction): State used to compute transition \
                rdm elements (optional).

            block2 (bool): Whether or not to use the block2 driver (optional).

        Returns:
            rdm (numpy.ndarray): The reduced density matrix.

        Raises:
            ValueError: If provided string corresponds to beyond a full 3pdm.
        """
        block2 = _default_fqe_opts["block2"] if block2 is None else block2

        # check the rank of the operator
        rank = len(string.split()) // 2
        if len(string.split()) % 2 != 0:
            raise ValueError("RDM must have even number of operators.")

        # Get an individual rdm element
        if any(char.isdigit() for char in string):
            mpo = mpo_from_fqe_hamiltonian(
                SparseHamiltonian(FermionOperator(string)),
                n_sites=self.n_sites)
            return self.expectationValue(mpo, brawfn)

        if block2:
            if brawfn is not None:
                raise ValueError(
                    "Transition density not implemented with block2 driver.")
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
        try:
            n_threads = int(os.environ['OMP_NUM_THREADS'])
        except KeyError:
            # OMP_NUM_THREADS is not set
            n_threads = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ['TMPDIR'] = str(temp_dir)
            driver = DMRGDriver(scratch=os.environ['TMPDIR'],
                                symm_type=SymmetryTypes.SZ | SymmetryTypes.CPX,
                                n_threads=n_threads)
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

    def _block2_tddmrg(self, time: float, hamiltonian: MPS, **kwargs):
        # Gather all options
        steps = kwargs.get("steps", _default_fqe_opts["steps"])
        n_sub_sweeps = kwargs.get("n_sub_sweeps",
                                  _default_fqe_opts["n_sub_sweeps"])
        add_noise = kwargs.get("add_noise", _default_fqe_opts["add_noise"])
        noise = 1e-18

        bdim = self.opts["max_bond_dim"]
        cutoff = kwargs.get("cutoff", _default_pyblock_opts["cutoff"])

        iprint = kwargs.get("iprint", 0)
        normalize = kwargs.get("normalize", False)

        dt = time / steps
        # Make MPS complex if not already and if doing real time propagation
        if not numpy.iscomplexobj(self.tensors[0].data):
            for i in range(self.n_sites):
                self.tensors[i].data = self.tensors[i].data.astype(complex)

        # Avoid Pade issue by adding a negligible noise term
        if add_noise:
            hamiltonian = MPS(tensors=hamiltonian.tensors,
                              opts=hamiltonian.opts,
                              const=noise)

        if bdim == -1:
            bdim = 4 ** ((self.n_sites + 1) // 2)
        try:
            n_threads = int(os.environ['OMP_NUM_THREADS'])
        except KeyError:
            # OMP_NUM_THREADS is not set
            n_threads = 1

        with tempfile.TemporaryDirectory() as temp_dir:
            os.environ['TMPDIR'] = str(temp_dir)
            driver = DMRGDriver(scratch=os.environ['TMPDIR'],
                                symm_type=SymmetryTypes.SZ | SymmetryTypes.CPX,
                                n_threads=n_threads)
            driver.initialize_system(n_sites=self.n_sites,
                                     orb_sym=[0]*self.n_sites)
            b2mps = MPSTools.to_block2(self, save_dir=driver.scratch)
            b2mpo = MPOTools.to_block2(hamiltonian, use_complex=True)

            b2mps = driver.td_dmrg(b2mpo, b2mps, delta_t=dt*1j, n_steps=steps,
                                   bond_dims=[bdim],
                                   n_sub_sweeps=n_sub_sweeps,
                                   cutoff=cutoff, iprint=iprint,
                                   normalize_mps=normalize)
            mps = MPSTools.from_block2(b2mps).to_flat()
            if add_noise:
                mps = mps*numpy.exp(1j*dt*noise)

        return type(self)(tensors=mps.tensors, opts=self.opts)


def get_hf_mps(nele, sz, norbs, bdim,
               e0: float = 0.0,
               cutoff: float = 0.0,
               full: bool = True,
               occ: Optional[list[int]] = None,
               dtype: type = float) -> "MPSWavefunction":
    """ Returns a Hartree-Fock MPSWavefunction.

    Args:
        nele (int): Number of electrons.

        sz (int): Total spin.

        norbs (int): Number of spatial orbitals.

        bdim (int): Maximum bond dimension for the resulting MPSWavefunction.

        e0 (float): Reference energy for the state.

        cutoff (float): Cutoff for the resulting MPSWavefunction.

        full (bool): Whether or not to include full space.

        occ (list(int)): Lost of orbital occupation numbers.

        dtype (type): data type for MPS tensors.

    Returns:
        wfn (MPSWavefunction): MPSWavefunction corresponding to a HF state.

    Raises:
        ValueError: If the number of electrons, total spin, and number of \
            orbitals are not compatible with each other, or if the provided \
            occupancy is not compatible with these.
    """
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
            f"Inconsistent doubly occupied orbitals: {occd} ({ndocc})")
    if occs != nsocc:
        raise ValueError(
            f"Inconsistent singly occupied orbitals: {occs} ({nsocc})")
    if occv != nvirt:
        raise ValueError(
            f"Inconsistent virtual orbitals: {occv} ({nvirt})")

    hamil = Hamiltonian(fd, flat=True)
    mps_info = MPSInfo(hamil.n_sites, hamil.vacuum, hamil.target, hamil.basis)
    mps_info.set_bond_dimension_occ(bdim, occ=occ)
    mps_wfn = MPS.ones(mps_info, dtype=dtype)
    if full:
        mps_info_full = MPSInfo(
            hamil.n_sites, hamil.vacuum, hamil.target, hamil.basis)
        mps_info_full.set_bond_dimension(bdim)
        mps_wfn += 0*MPS.ones(mps_info_full)
    return MPSWavefunction.from_pyblock3_mps(mps_wfn, max_bond_dim=bdim,
                                             cutoff=cutoff)


def get_random_mps(nele, sz, norbs, bdim,
                   e0: float = 0.0,
                   cutoff: float = 0.0,
                   dtype: type = float) -> "MPSWavefunction":
    """ Returns a random MPSWavefunction.

    Args:
        nele (int): Number of electrons.

        sz (int): Total spin.

        norbs (int): Number of spatial orbitals.

        bdim (int): Maximum bond dimension for the resulting MPSWavefunction.

        e0 (float): Reference energy for the state.

        cutoff (float): Cutoff for the resulting MPSWavefunction.

        dtype (type): data type for MPS tensors.

    Returns:
        wfn (MPSWavefunction): MPSWavefunction corresponding to a random state.

    Raises:
        ValueError: If the number of electrons, total spin, and number of \
            orbitals are not compatible with each other, or if the provided \
            occupancy is not compatible with these.
    """
    if (nele + abs(sz)) // 2 > norbs:
        raise ValueError(
            f"Electron number is too large (nele = {nele}, norb = {norbs})")
    if sz % 2 != nele % 2:
        raise ValueError(
            f"Spin (sz = {sz}) is incompatible with nele = {nele}")

    nsocc = abs(sz)
    ndocc = (nele - nsocc) // 2
    nvirt = norbs - nsocc - ndocc
    assert nvirt >= 0

    fd = FCIDUMP(pg='c1',
                 n_sites=norbs,
                 const_e=e0,
                 n_elec=nele,
                 twos=sz)
    hamil = Hamiltonian(fd, flat=True)
    mps_info = MPSInfo(hamil.n_sites, hamil.vacuum, hamil.target, hamil.basis)
    mps_info.set_bond_dimension(bdim)
    mps_wfn = MPS.random(mps_info, dtype=dtype)
    return MPSWavefunction.from_pyblock3_mps(mps_wfn, max_bond_dim=bdim,
                                             cutoff=cutoff)


_default_fqe_opts = {
    "steps": 1,
    "n_sub_sweeps": 1,
    "method": "tddmrg",
    "cached": False,
    "block2": True,
    "add_noise": False
}


_default_pyblock_opts = {
    "cutoff": 1e-14,
    "max_bond_dim": -1
}
