import itertools

import numpy
from fqe.fqe_data import FqeData
from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP


def fqe_sign_change(fqedata: FqeData) -> numpy.ndarray:
    """Function to return the fermionic sign change.

    Args:
        fqedata (FqeData): Openfermion-FQE FqeData object.

    Returns:
        fermionic_sign (numpy.ndarray): Sign changes for states.
    """
    ndim = fqedata.norb()
    alpha_str = fqedata.get_fcigraph().string_alpha_all().astype(int)
    beta_str = fqedata.get_fcigraph().string_beta_all().astype(int)
    alpha_occ = numpy.array([tuple(reversed(bin(1 << ndim | x)[3:]))
                             for x in alpha_str], dtype=numpy.uint32)
    beta_occ = numpy.array([tuple(reversed(bin(1 << ndim | x)[3:]))
                            for x in beta_str], dtype=numpy.uint32)

    cumsum_alpha_occ = numpy.cumsum(alpha_occ, axis=1)
    n_alpha, n_beta = sum(alpha_occ[0]), sum(beta_occ[0])
    swaps = cumsum_alpha_occ @ beta_occ.T + (n_alpha * n_beta)
    fermionic_sign = numpy.where(swaps % 2 == 0, 1.0, -1.0)
    return fermionic_sign


def one_body_projection_mpo(isite: int, jsite: int, n_sites: int,
                            flat: bool = True, spinfree: bool = True):
    """Function to return a one body projection operator MPO, |isite><jsite|.

    Args:
        isite (int): Site index for the ket.

        jsite (int): Site index for the bra.

        n_sites (int): Total number of spatial sites.

        flat (bool): Whether to use pyblock3 flat machinery.

        spinfree (bool): Whether to sum over spin orbitals at provided sites.

    Returns:
        projection_operator (MPS): MPO for the one-body projection operator.
    """
    fd = FCIDUMP(pg="c1", n_sites=n_sites)

    def gen_spinfree_terms(n_sites, c, d):
        for sigma in [0, 1]:
            yield c[isite, sigma] * d[jsite, sigma]

    def gen_spin_terms(n_sites, c, d):
        yield c[isite // 2, isite % 2] * d[jsite // 2, jsite % 2]

    hamil = Hamiltonian(fd, flat)
    if spinfree:
        mpo = hamil.build_mpo(gen_spinfree_terms,
                              cutoff=1E-12,
                              max_bond_dim=-1,
                              const=0)
    else:
        mpo = hamil.build_mpo(gen_spin_terms,
                              cutoff=1E-12,
                              max_bond_dim=-1,
                              const=0)

    return mpo.to_sparse()


def two_body_projection_mpo(isite: int, jsite: int, ksite: int, lsite: int,
                            n_sites: int, flat: bool = True,
                            spinfree: bool = True):
    """Function to return a two body projection operator MPO, \
    |isite, jsite><ksite, lsite|.

    Args:
        isite (int): First site index for the ket.

        jsite (int): Second site index for the ket.

        ksite (int): First site index for the bra.

        lsite (int): Second site index for the bra.

        n_sites (int): Total number of spatial sites.

        flat (bool): Whether to use pyblock3 flat machinery.

        spinfree (bool): Whether to sum over spin orbitals at provided sites.

    Returns:
        projection_operator (MPS): MPO for the two-body projection operator.
    """
    fd = FCIDUMP(pg="c1", n_sites=n_sites)

    def gen_spin_terms(n_sites, c, d):
        yield c[isite // 2, isite % 2] * c[jsite // 2, jsite % 2] \
            * d[ksite // 2, ksite % 2] * d[lsite // 2, lsite % 2]

    def gen_spinfree_terms(n_sites, c, d):
        for sigma in [0, 1]:
            for rho in [0, 1]:
                yield c[isite, sigma] * c[jsite, rho] \
                    * d[ksite, sigma] * d[lsite, rho]

    hamil = Hamiltonian(fd, flat)
    if spinfree:
        mpo = hamil.build_mpo(gen_spinfree_terms, cutoff=1E-12,
                              max_bond_dim=-1, const=0)
    else:
        mpo = hamil.build_mpo(gen_spin_terms, cutoff=1E-12,
                              max_bond_dim=-1, const=0)

    return mpo.to_sparse()


def three_body_projection_mpo(isite: int, jsite: int, ksite: int,
                              lsite: int, msite: int, nsite: int,
                              n_sites: int, flat: bool = True,
                              spinfree: bool = True):
    """Function to return a two body projection operator MPO, \
    |isite, jsite><ksite, lsite|.

    Args:
        isite (int): First site index for the ket.

        jsite (int): Second site index for the ket.

        ksite (int): Third site index for the ket.

        lsite (int): First site index for the bra.

        msite (int): Second site index for the bra.

        nsite (int): Third site index for the bra.

        n_sites (int): Total number of spatial sites.

        flat (bool): Whether to use pyblock3 flat machinery.

        spinfree (bool): Whether to sum over spin orbitals at provided sites.

    Returns:
        projection_operator (MPS): MPO for the three-body projection operator.
    """
    fd = FCIDUMP(pg="c1", n_sites=n_sites)

    def gen_spin_terms(n_sites, c, d):
        yield c[isite // 2, isite % 2] * c[jsite // 2, jsite % 2] \
            * c[ksite // 2, ksite % 2] * d[lsite // 2, lsite % 2] \
            * d[msite // 2, msite % 2] * d[nsite // 2, nsite % 2]

    def gen_spinfree_terms(n_sites, c, d):
        for sigma, rho, tau in itertools.product([0, 1], repeat=3):
            yield c[isite, sigma] * c[jsite, rho] * c[ksite, tau]\
                * d[lsite, sigma] * d[msite, rho] * d[nsite, tau]

    hamil = Hamiltonian(fd, flat)
    if spinfree:
        mpo = hamil.build_mpo(gen_spinfree_terms, cutoff=1E-12,
                              max_bond_dim=-1, const=0)
    else:
        mpo = hamil.build_mpo(gen_spin_terms, cutoff=1E-12,
                              max_bond_dim=-1, const=0)

    return mpo.to_sparse()


def apply_fiedler_ordering(h1, h2):
    """Reorder orbitals using the Fiedler method.

    Args:
        h1 (numpy.ndarray): One-electron intergral array.

        h2 (numpy.ndarray): Two-electron intergral array.

    Returns:
        h1new (numpy.ndarray): Re-ordered one-electron intergral array.

        h2new (numpy.ndarray): Re-ordered two-electron intergral array.

        order (numpy.ndarray): Order of orbitals.
    """

    def fiedler_order(mat):
        n = mat.shape[0]

        # Laplacian matrix of the graph
        lmat = numpy.zeros(mat.shape)
        for i in range(n):
            for j in range(n):
                lmat[i, i] += abs(mat[i, j])
                lmat[i, j] -= mat[i, j]

        ee, ev = numpy.linalg.eigh(lmat)
        assert abs(ee[0] < 1E-12)
        factor = 1.0
        for x in ev[1]:
            if abs(x) > 1E-12:
                factor = 1 if x > 0 else -1
                break

        sort_key = factor*ev[1]
        order = list(range(n))
        return sorted(order, key=lambda x: sort_key[x])

    # use the K-matrix but break any unwanted symmetries
    # by perturbing with h1
    mat = 1E-6*h1 + numpy.einsum('ijji->ij', h2)
    order = fiedler_order(mat)
    # reorder indices
    h1_new = h1[:, order]
    h1_new = h1_new[order]

    h2_new = h2[:, :, :, order]
    h2_new = h2_new[:, :, order]
    h2_new = h2_new[:, order]
    h2_new = h2_new[order]

    return h1_new, h2_new, order
