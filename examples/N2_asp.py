from functools import singledispatch

import numpy
from pyscf import ao2mo, gto, scf
import fqe
from mps_fqe.wavefunction import MPSWavefunction, get_hf_mps


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


def asp(h1e, g2e, fock, wfn, ns, dt):
    """Adiabatic state preparation with linear schedule."""
    hamil = fqe.get_restricted_hamiltonian(
        (h1e, numpy.einsum("ikjl", -0.5 * g2e)), e_0=e0)
    energies = []
    max_time = ns*dt
    for i in range(ns):
        t = (i + 1)*dt
        c1 = t/max_time
        c2 = (max_time - t)/max_time
        hs = fqe.get_restricted_hamiltonian(
            (c1*h1e + c2*fock, numpy.einsum("ikjl", -c1*0.5 * g2e)), e_0=e0)
        wfn = wfn.time_evolve(dt, hs)
        norm = numpy.sqrt(vdot(wfn, wfn))
        wfn.scale(1./norm)
        energies.append(wfn.expectationValue(hamil).real)
    return numpy.asarray(energies)


if __name__ == '__main__':
    bond_dist = 1.5
    nsteps = 100
    stepsize = 0.1
    bdim = 100
    basis = 'sto-6g'
    outfile = "N2_asp.dat"
    mf = N2_from_pyscf(bond_dist, basis)

    nele = mf.mol.nelectron
    sz = mf.mol.spin
    e0 = mf.mol.energy_nuc()

    norb = mf.mo_coeff.shape[1]
    fock = mf.mo_coeff.T @ mf.get_fock() @ mf.mo_coeff
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    g2e = ao2mo.restore(1, ao2mo.kernel(mf.mol, mf.mo_coeff),
                        norb)
    hamil = fqe.get_restricted_hamiltonian(
        (h1e, numpy.einsum("ikjl", -0.5 * g2e)), e_0=e0)
    print(f"Problem size (nele, norb): ({nele}, {norb})")

    # Use FQE to do the adiabatic state prep (exact evolution)
    init_wf = fqe.Wavefunction([[nele, 0, norb]])
    init_wf.set_wfn(strategy='hartree-fock')  # HF initial state
    out_fqe = asp(h1e, g2e, fock, init_wf, nsteps, stepsize)

    # Use MPS-FQE to do the adiabatic state prep (truncated bond dimension)
    mps = get_hf_mps(nele, sz, norb, bdim, cutoff=1E-12, full=True)
    out_mps = asp(h1e, g2e, fock, mps, nsteps, stepsize)

    with open(outfile, 'w') as f:
        f.write(f"# System: N2, {basis}, R = {bond_dist}\n")
        f.write(f"# t, E(exact),  E(bdim={bdim})\n")
        t = stepsize
        for e_fqe, e_mps in zip(out_fqe, out_mps):
            f.write(f"{t:5.2f} {e_fqe:10.5f} {e_mps:10.5f}\n")
            t += stepsize
