import numpy

from fqe import Wavefunction as FqeWavefunction
from fqe.hamiltonians.hamiltonian import Hamiltonian as FqeHamiltonian
from fqe.hamiltonians import diagonal_coulomb, diagonal_hamiltonian, \
                             general_hamiltonian, gso_hamiltonian, \
                             hamiltonian, restricted_hamiltonian, \
                             sparse_hamiltonian, sso_hamiltonian

from pyblock3.hamiltonian import Hamiltonian
from pyblock3.fcidump import FCIDUMP
from pyblock3.algebra.mps import MPS


allowed_types = [
    sparse_hamiltonian.SparseHamiltonian,
    diagonal_coulomb.DiagonalCoulomb,
    diagonal_hamiltonian.Diagonal,
    restricted_hamiltonian.RestrictedHamiltonian
]

class MPOHamiltonian(Hamiltonian):
    @classmethod
    def from_fqe_hamiltonian(cls, fqe_wfn: FqeWavefunction,
                             fqe_ham: FqeHamiltonian,
                             pg="c1",
                             flat=True,
                             cutoff=1E-9) -> "MPS":
        if not fqe_wfn.conserve_number():
            raise TypeError('Wavefunction does not conserve number of particles.')
        if not fqe_wfn.conserve_spin():
            raise TypeError('Wavefunction does not conserve spin.')
        fd = FCIDUMP(pg=pg,
                     n_sites=fqe_wfn.norb(),
                     n_elec=fqe_wfn._conserved['n'],
                     twos=fqe_wfn._conserved['s_z'],
                     const_e=fqe_ham.e_0())

        for typ in allowed_types:
            if isinstance(fqe_ham, typ):
                return hamiltonian_function_dict[typ](fqe_ham, fd, flat)        
        raise TypeError("Have not implemented MPO for {}".format(type(fqe_ham)))

def _get_restricted_mpo(fqe_ham, fd, flat):
    #generate the restricted hamiltonian MPO
    hamil = Hamiltonian(fd, flat=flat)
    def generate_terms(n_sites, c, d):
        t = fqe_ham.tensors()[0]
        v = numpy.einsum("ikjl", -2 * fqe_ham.tensors()[1])
        for isite in range(0, n_sites):
            for jsite in range(0, n_sites):
                for ispin in [0, 1]:
                    yield t[isite, jsite] * c[isite, ispin] * d[jsite, ispin]
        for isite in range(0, n_sites):
            for jsite in range(0, n_sites):
                for ksite in range(0, n_sites):
                    for lsite in range(0, n_sites):
                        for ijspin in [0, 1]:
                            for klspin in [0, 1]:
                                yield 0.5*v[isite, jsite, ksite, lsite] \
                                    * (c[isite, ijspin] * c[ksite, klspin]
                                       * d[lsite, klspin] * d[jsite, ijspin])
    return hamil.build_mpo(generate_terms, const=fqe_ham.e_0(), cutoff=0).to_sparse()

def _get_diagonal_coulomb_mpo(fqe_ham, fd, flat):
    #generate the diagonal coulomb MPO
    hamil = Hamiltonian(fd, flat=flat)
    def generate_terms(n_sites, c, d):
        v = fqe_ham._tensor[2]
        for isite in range(0, n_sites):
            for jsite in range(0, n_sites):
                for ispin in [0, 1]:
                    for jspin in [0, 1]:
                        yield v[isite, jsite] \
                            * (c[isite, ispin] * c[jsite, jspin]
                               * d[jsite, jspin] * d[isite, ispin])
    return hamil.build_mpo(generate_terms, const=fqe_ham.e_0(), cutoff=0).to_sparse()

def _get_diagonal_mpo(fqe_ham, fd, flat):
    #generate the diagonal MPO
    hamil = Hamiltonian(fd, flat=flat)
    t = fqe_ham.diag_values()
    def generate_terms(n_sites, c, d):
        for isite in range(0, n_sites):
            for ispin in [0, 1]:
                yield t[isite] * (c[isite, ispin] * d[isite, ispin])
    return hamil.build_mpo(generate_terms, const=fqe_ham.e_0(), cutoff=0).to_sparse()

def _get_sparse_mpo(fqe_ham, fd, flat):
    #generate the sparse Hamiltonian MPO
    hamil = Hamiltonian(fd, flat=flat)
    def generate_terms(n_sites, c, d):
        #Define mapping between representationas
        def _operator_map(d_c_ops=[d, c], alpha_terms=None, beta_terms=None):
            alpha_prod = [1]*2
            beta_prod = [1]*2
            for at in alpha_terms:
                alpha_prod[at[1]] *= d_c_ops[at[1]][at[0],0]
            for bt in beta_terms:
                beta_prod[bt[1]] *= d_c_ops[bt[1]][bt[0],1]
            return {'alpha': alpha_prod[1] * alpha_prod[0],
                    'beta': beta_prod[1] * beta_prod[0] }

        for term in fqe_ham.terms():
            mpo_operators = _operator_map([d, c], alpha_terms=term[1], beta_terms=term[2])
            yield term[0] * mpo_operators["alpha"] * mpo_operators["beta"]
            
    return hamil.build_mpo(generate_terms, const=fqe_ham.e_0(), cutoff=0).to_sparse()

hamiltonian_function_dict = {
    sparse_hamiltonian.SparseHamiltonian: _get_sparse_mpo,
    diagonal_coulomb.DiagonalCoulomb: _get_diagonal_coulomb_mpo,
    diagonal_hamiltonian.Diagonal: _get_diagonal_mpo,
    restricted_hamiltonian.RestrictedHamiltonian: _get_restricted_mpo
}
