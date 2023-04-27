import numpy

from fqe.fqe_data import FqeData
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP


_spin_value = {"alpha": 0,
               "beta": 1}


def fqe_sign_change(fqedata: FqeData) -> numpy.ndarray:
    ndim = fqedata.norb()
    alpha_str = fqedata.get_fcigraph().string_alpha_all().astype(int)
    beta_str = fqedata.get_fcigraph().string_beta_all().astype(int)
    alpha_occ = numpy.array([tuple(reversed(bin(1 << ndim | x)[3:]))
                             for x in alpha_str], dtype=numpy.uint32)
    beta_occ = numpy.array([tuple(reversed(bin(1 << ndim | x)[3:]))
                            for x in beta_str], dtype=numpy.uint32)

    cumsum_alpha_occ = numpy.cumsum(alpha_occ, axis=1)
    n_alpha, n_beta = sum(alpha_occ[0]), sum(beta_occ[0])
    swaps = numpy.sum(cumsum_alpha_occ[:, None, :]
                      * beta_occ[None, :, :], axis=2) + (n_alpha * n_beta)
    fermionic_sign = numpy.where(swaps % 2 == 0, 1.0, -1.0)
    return fermionic_sign


def one_body_projection_mpo(isite: int, jsite: int, n_sites: int,
                            flat: bool = True, spinfree: bool = True):
    fd = FCIDUMP(pg="c1", n_sites=n_sites)

    def gen_spinfree_terms(n_sites, c, d):
        for sigma in [0, 1]:
            yield c[isite, sigma] * d[jsite, sigma]

    def gen_spin_terms(n_sites, c, d):
        yield c[isite // 2, isite % 2] * d[jsite // 2, jsite % 2]

    if spinfree:
        mpo = Hamiltonian(fd, flat).build_mpo(gen_spinfree_terms,
                                              cutoff=1E-12,
                                              max_bond_dim=-1)
    else:
        mpo = Hamiltonian(fd, flat).build_mpo(gen_spin_terms,
                                              cutoff=1E-12,
                                              max_bond_dim=-1)
    return mpo.to_sparse()


def two_body_projection_mpo(isite: int, jsite: int, ksite: int, lsite: int,
                            n_sites: int, flat: bool = True,
                            spinfree: bool = True):
    fd = FCIDUMP(pg="c1", n_sites=n_sites)

    def gen_spin_terms(n_sites, c, d):
        yield c[isite // 2, isite % 2] * c[jsite // 2, jsite % 2] \
            * d[lsite // 2, lsite % 2] * d[ksite // 2, ksite % 2]

    def gen_spinfree_terms(n_sites, c, d):
        for sigma in [0, 1]:
            for tau in [0, 1]:
                yield c[isite, sigma] * c[jsite, tau] \
                    * d[ksite, sigma] * d[lsite, tau]

    if spinfree:
        mpo = Hamiltonian(fd, flat).build_mpo(gen_spinfree_terms,
                                              cutoff=0,
                                              max_bond_dim=-1)
    else:
        mpo = Hamiltonian(fd, flat).build_mpo(gen_spin_terms,
                                              cutoff=0,
                                              max_bond_dim=-1)
    return mpo.to_sparse()
