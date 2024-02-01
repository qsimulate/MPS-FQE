from functools import singledispatch

import numpy
from pyscf import ao2mo, gto, scf
import fqe
from mps_fqe.wavefunction import MPSWavefunction, get_hf_mps
from mps_fqe.hamiltonian import mpo_from_fqe_hamiltonian


def N2_from_pyscf(r, basis):
    geom = "N 0.0 0.0 0.0\n"
    geom += f"N 0.0 0.0 {r}"
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol_grad = 1e-6
    mf.conv_tol = 1e-10
    energy = mf.kernel()
    print(f"SCF energy: {energy}")
    return mf


@singledispatch
def vdot(bra, ket):
    return fqe.vdot(bra, ket)


@vdot.register(MPSWavefunction)
def _(bra, ket):
    return numpy.dot(numpy.conj(bra), ket)


def imag(hamil, wfn, ns, dt):
    energies = []
    for i in range(ns):
        wfn = wfn.time_evolve(-1j * dt, hamil)
        norm = numpy.sqrt(vdot(wfn, wfn))
        wfn.scale(1./norm)
        energies.append(wfn.expectationValue(hamil).real)
    return numpy.asarray(energies)


if __name__ == '__main__':
    bond_dist = 1.5
    nsteps = 50
    stepsize = 0.1
    bdim = 50
    basis = 'sto-6g'
    outfile = "N2_imag.dat"
    mf = N2_from_pyscf(bond_dist, basis)

    nele = mf.mol.nelectron
    sz = mf.mol.spin
    e0 = mf.mol.energy_nuc()

    norb = mf.mo_coeff.shape[1]
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    g2e = ao2mo.restore(1, ao2mo.kernel(mf.mol, mf.mo_coeff),
                        norb)
    hamil = fqe.get_restricted_hamiltonian(
        (h1e, numpy.einsum("ikjl", -0.5 * g2e)), e_0=e0)
    print(f"Problem size (nele, norb): ({nele}, {norb})")

    # Use FQE to do the imaginary time evolution (exact)
    init_wf = fqe.Wavefunction([[nele, 0, norb]])
    init_wf.set_wfn(strategy='hartree-fock')  # HF initial state
    out_fqe = imag(hamil, init_wf, nsteps, stepsize)
    print(out_fqe)

    # Use MPS-FQE to do the imaginary time evolution (truncated bond dimension)
    mpo = mpo_from_fqe_hamiltonian(hamil)
    mps = get_hf_mps(nele, sz, norb, bdim, cutoff=1E-50, full=True)
    out_mps = imag(mpo, mps, nsteps, stepsize)
    print(out_mps)

    with open(outfile, 'w') as f:
        f.write(f"# System: N2, {basis}, R = {bond_dist}\n")
        f.write(f"# t, E(exact),  E(bdim={bdim})\n")
        t = stepsize
        for e_fqe, e_mps in zip(out_fqe, out_mps):
            f.write(f"{t:4.2f} {e_fqe:10.5f} {e_mps:10.5f}\n")
            t += stepsize
