import numpy
from typing import Optional

from fqe.hamiltonians.hamiltonian import Hamiltonian as FqeHamiltonian
from fqe.hamiltonians import diagonal_coulomb, diagonal_hamiltonian, \
    restricted_hamiltonian, sparse_hamiltonian, general_hamiltonian

from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.algebra.mps import MPS


allowed_types = [
    sparse_hamiltonian.SparseHamiltonian,
    diagonal_coulomb.DiagonalCoulomb,
    diagonal_hamiltonian.Diagonal,
    restricted_hamiltonian.RestrictedHamiltonian
]


def mpo_from_fqe_hamiltonian(fqe_ham: FqeHamiltonian,
                             n_sites: Optional[int] = None,
                             pg: str = "c1",
                             flat: bool = True,
                             cutoff: float = 1E-12,
                             max_bond_dim: int = -1) -> "MPS":
    if n_sites is None:
        if isinstance(fqe_ham, sparse_hamiltonian.SparseHamiltonian):
            raise ValueError("Must provide n_sites for sparse Hamiltonian")
        else:
            n_sites = fqe_ham.dim()

    fd = FCIDUMP(pg=pg,
                 n_sites=n_sites,
                 const_e=fqe_ham.e_0())

    for typ in allowed_types:
        if isinstance(fqe_ham, typ):
            return _hamiltonian_func_dict[typ](fqe_ham,
                                               fd,
                                               flat,
                                               cutoff,
                                               max_bond_dim)
    raise TypeError(f"Have not implemented MPO for {type(fqe_ham)}")


def get_sparse_mpo(fqe_ham: FqeHamiltonian,
                   fd: FCIDUMP,
                   flat: bool = True,
                   cutoff: float = 1E-12,
                   max_bond_dim: int = -1) -> "MPS":
    # Generate the sparse Hamiltonian MPO
    hamil = Hamiltonian(fd, flat=flat)

    def generate_terms(n_sites, c, d):
        # Define mapping between representationas
        def _operator_map(d_c_ops, alpha_terms, beta_terms):
            alpha_prod = [1] * 2
            beta_prod = [1] * 2
            for a0, a1 in alpha_terms:
                alpha_prod[a1] *= d_c_ops[a1][a0, 0]
            for b0, b1 in beta_terms:
                beta_prod[b1] *= d_c_ops[b1][b0, 1]
            return {'alpha': alpha_prod[1] * alpha_prod[0],
                    'beta': beta_prod[1] * beta_prod[0]}

        for (coeff, alpha_terms, beta_terms) in fqe_ham.terms():
            mpo_operators = _operator_map([d, c],
                                          alpha_terms,
                                          beta_terms)
            yield coeff * mpo_operators["alpha"] * mpo_operators["beta"]

    mpo = hamil.build_mpo(generate_terms,
                          const=fqe_ham.e_0(),
                          cutoff=cutoff,
                          max_bond_dim=max_bond_dim)

    return mpo.to_sparse()


def get_restricted_mpo(fqe_ham: FqeHamiltonian,
                       fd: FCIDUMP,
                       flat: bool = True,
                       cutoff: float = 1E-12,
                       max_bond_dim: int = -1) -> "MPS":
    # Generate the restricted hamiltonian MPO
    fd.h1e = fqe_ham.tensors()[0]
    fd.g2e = numpy.einsum("ikjl", -2 * fqe_ham.tensors()[1])
    hamil = Hamiltonian(fd, flat=flat)
    mpo, _ = hamil.build_qc_mpo().compress(cutoff=cutoff,
                                           max_bond_dim=max_bond_dim)
    opts = {'cutoff': cutoff,
            'max_bond_dim': max_bond_dim}

    return MPS(tensors=mpo.tensors, opts=opts, const=fd.const_e).to_sparse()


def get_complex_qc_mpo(fqe_ham: FqeHamiltonian,
                       fd: FCIDUMP,
                       flat: bool = True,
                       cutoff: float = 1E-12,
                       max_bond_dim: int = -1) -> "MPS":
    # Generate the restricted hamiltonian MPO
    fd.h1e = fqe_ham.tensors()[0]
    fd.g2e = numpy.einsum("ikjl", -2 * fqe_ham.tensors()[1])
    hamil = Hamiltonian(fd, flat=flat)
    mpo, _ = hamil.build_complex_qc_mpo().compress(cutoff=cutoff,
                                                   max_bond_dim=max_bond_dim)
    opts = {'cutoff': cutoff,
            'max_bond_dim': max_bond_dim}

    return MPS(tensors=mpo.tensors, opts=opts, const=fd.const_e).to_sparse()


def get_diagonal_coulomb_mpo(fqe_ham: FqeHamiltonian,
                             fd: FCIDUMP,
                             flat: bool = True,
                             cutoff: float = 1E-12,
                             max_bond_dim: int = -1) -> "MPS":
    # Generate the diagonal coulomb MPO
    hamil = Hamiltonian(fd, flat=flat)

    def generate_terms(n_sites, c, d):
        v = fqe_ham._tensor[2]
        diag = fqe_ham._tensor[1]
        for isite in range(n_sites):
            for ispin in [0, 1]:
                yield (diag[isite] + v[isite, isite])\
                    * c[isite, ispin] * d[isite, ispin]
        for isite in range(n_sites):
            for jsite in range(n_sites):
                for ispin in [0, 1]:
                    for jspin in [0, 1]:
                        yield v[isite, jsite] \
                            * (c[isite, ispin] * c[jsite, jspin]
                               * d[jsite, jspin] * d[isite, ispin])

    mpo = hamil.build_mpo(generate_terms,
                          const=fqe_ham.e_0(),
                          cutoff=cutoff,
                          max_bond_dim=max_bond_dim)

    return mpo.to_sparse()


def get_diagonal_mpo(fqe_ham: FqeHamiltonian,
                     fd: FCIDUMP,
                     flat: bool = True,
                     cutoff: float = 1E-12,
                     max_bond_dim: int = -1) -> "MPS":
    # Generate the diagonal MPO
    hamil = Hamiltonian(fd, flat=flat)
    t = fqe_ham.diag_values()

    def generate_terms(n_sites, c, d):
        for isite in range(n_sites):
            for ispin in [0, 1]:
                yield t[isite] * (c[isite, ispin] * d[isite, ispin])

    mpo = hamil.build_mpo(generate_terms,
                          const=fqe_ham.e_0(),
                          cutoff=cutoff,
                          max_bond_dim=max_bond_dim)

    return mpo.to_sparse()


_hamiltonian_func_dict = {
    sparse_hamiltonian.SparseHamiltonian: get_sparse_mpo,
    diagonal_coulomb.DiagonalCoulomb: get_diagonal_coulomb_mpo,
    diagonal_hamiltonian.Diagonal: get_diagonal_mpo,
    restricted_hamiltonian.RestrictedHamiltonian: get_complex_qc_mpo
    }
