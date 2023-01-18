import numpy as np
from fqe.fqe_data import FqeData


def fqe_sign_change(fqedata: FqeData) -> np.ndarray:
    ndim = fqedata.norb()
    alpha_str = fqedata.get_fcigraph().string_alpha_all().astype(int)
    beta_str = fqedata.get_fcigraph().string_beta_all().astype(int)
    alpha_occ = np.array([tuple(reversed(bin(1 << ndim | x)[3:])) for x in alpha_str], dtype=np.uint32)
    beta_occ = np.array([tuple(reversed(bin(1 << ndim | x)[3:])) for x in beta_str], dtype=np.uint32)

    cumsum_alpha_occ = np.cumsum(alpha_occ, axis=1)
    n_alpha, n_beta = sum(alpha_occ[0]), sum(beta_occ[0])
    swaps = np.sum(cumsum_alpha_occ[:, None, :] * beta_occ[None, :, :], axis=2) + (n_alpha * n_beta)
    fermionic_sign = np.where(swaps % 2 == 0, 1.0, -1.0)
    return fermionic_sign
