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

class MPOHamiltonian(Hamiltonian):
    @classmethod
    def from_fqe_specific(cls, fqe_wfn: FqeWavefunction,
                          fqe_ham: FqeHamiltonian,
                          pg="c1",
                          flat=False) -> "MPS":
        if not fqe_wfn.conserve_number():
            raise TypeError('Wavefunction does not conserve number of particles.')
        if not fqe_wfn.conserve_spin():
            raise TypeError('Wavefunction does not conserve spin.')
        #RestrictedHamiltonian
        if isinstance(fqe_ham, restricted_hamiltonian.RestrictedHamiltonian):
            h1e = fqe_ham.tensors()[0]
            g2e = numpy.einsum("ikjl", -2*fqe_ham.tensors()[1])
            fd = FCIDUMP(pg=pg,
                         n_sites=fqe_wfn.norb(),
                         n_elec=fqe_wfn._conserved['n'],
                         twos=fqe_wfn._conserved['s_z'],
                         h1e=h1e,
                         g2e=g2e,
                         const_e=fqe_ham.e_0()
            )
            return cls(fcidump=fd, flat=flat).build_qc_mpo()
        else:
            raise TypeError('Have not implemented this type of Hamiltonian yet.')

    @classmethod
    def from_fqe_hamiltonian(cls, fqe_wfn: FqeWavefunction,
                             fqe_ham: FqeHamiltonian,
                             pg="c1",
                             flat=False,
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
        # If RestrictedHamiltonian
        if isinstance(fqe_ham, restricted_hamiltonian.RestrictedHamiltonian):
            return _get_restricted_ham_mpo(fqe_ham, fd, flat=flat)
        elif isinstance(fqe_ham, diagonal_coulomb.DiagonalCoulomb):
            return _get_diagonal_coulomb_mpo(fqe_ham, fd, flat=flat)

def _get_restricted_ham_mpo(fqe_ham, fd, flat):
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

