import numpy as np
from pyscf import gto, scf, lo, ao2mo
from pyblock3.fcidump import FCIDUMP
from pyblock3.algebra.mpe import MPE
from pyblock3.algebra.mps import MPS, MPSInfo
from pyblock3.hamiltonian import Hamiltonian
from mps_fqe.wavefunction import MPSWavefunction
from mps_fqe.utils import apply_fiedler_ordering


def get_geom(r):
    hydrogens = 10
    y_offset = np.sin(np.pi / 3.0) * r
    hx, hy, hz = [0]*hydrogens, [0]*hydrogens, [0]*hydrogens
    hx[0] = hx[7] = -r
    hx[2] = hx[9] = r
    hx[4] = -0.5*r
    hx[5] = 0.5*r
    hx[3] = -1.5*r
    hx[6] = 1.5*r
    hy[:3] = [y_offset]*3
    hy[7:] = [-y_offset]*3

    geom = ""
    for i in range(hydrogens):
        geom += "{}\t{}\t{}\t{}\n".format("H", hx[i], hy[i], hz[i])

    return geom


def localize_orbitals(pyscf_mf, method='pipek-mezey'):
    if method == 'pipek-mezey':
        mo_occ = pyscf_mf.mo_occ
        mo_coeff = pyscf_mf.mo_coeff
        docc_idx = np.where(np.isclose(mo_occ, 2.))[0]
        socc_idx = np.where(np.isclose(mo_occ, 1.))[0]
        virt_idx = np.where(np.isclose(mo_occ, 0.))[0]

        # Pipek-Mezey localization
        loc_docc_mo = lo.PM(pyscf_mf.mol,
                            mo_coeff[:, docc_idx]).kernel(verbose=5)
        loc_socc_mo = lo.PM(pyscf_mf.mol,
                            mo_coeff[:, socc_idx]).kernel(verbose=5)
        loc_virt_mo = lo.PM(pyscf_mf.mol,
                            mo_coeff[:, virt_idx]).kernel(verbose=5)

        loc_mo_coeff = np.hstack((loc_docc_mo, loc_socc_mo, loc_virt_mo))
        loc_mo_coeff = np.hstack((loc_docc_mo, loc_virt_mo))
        pyscf_mf.mo_coeff = loc_mo_coeff
    elif method == 'meta-lowdin':
        pyscf_mf.mo_coeff = lo.orth_ao(pyscf_mf.mol, method='meta_lowdin')
    else:
        raise ValueError("Localization method must \
        'pipek-mezey' or 'meta-lowdin'")


def get_hf_mps(nele, sz, norbs, bdim, e0=0, occ=None):
    fd = FCIDUMP(pg='c1',
                 n_sites=norbs,
                 const_e=e0,
                 n_elec=nele,
                 twos=sz)
    assert sz == 0  # only works for restricted for now
    if occ is None:
        occ = [2 if i < nele//2 else 0 for i in range(norbs)]
    hamil = Hamiltonian(fd, flat=True)
    mps_info = MPSInfo(hamil.n_sites, hamil.vacuum, hamil.target, hamil.basis)
    mps_info.set_bond_dimension_occ(bdim, occ=occ)
    mps_wfn = MPS.ones(mps_info)
    return MPSWavefunction.from_pyblock3_mps(mps_wfn, max_bond_dim=bdim)


def get_energy(mps, mpo, bdim):
    dmrg = MPE(mps, mpo, mps).dmrg(bdims=[bdim],
                                   noises=[1E-6, 0],
                                   dav_thrds=[1E-3],
                                   iprint=2,
                                   n_sweeps=10)
    E = dmrg.energies[-1].real
    return E


if __name__ == '__main__':
    r = 1.5
    geom = get_geom(r)
    mol = gto.M(atom=geom, basis='sto6g', verbose=0)
    mf = scf.RHF(mol)
    ener = mf.kernel()

    nele = mol.nelectron
    sz = mol.spin
    e0 = mol.energy_nuc()

    norbs = mf.mo_coeff.shape[1]
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    g2e = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff),
                        norbs)

    # Get original and reference HF MPS and MPO
    fd = FCIDUMP(pg='c1', n_sites=norbs, n_elec=nele, twos=sz, ipg=0,
                 uhf=False, h1e=h1e, g2e=g2e, const_e=e0)
    opts = {'cutoff': 1E-16,
            'max_bond_dim': -1}
    mpo_original = Hamiltonian(fd, flat=True).build_qc_mpo()
    mpo_original = MPS(tensors=mpo_original.tensors, const=fd.const_e,
                       opts=opts).to_sparse()
    # Get Reference energy
    mps_reference = get_hf_mps(nele=nele,
                               sz=sz,
                               norbs=norbs,
                               bdim=-1,
                               e0=e0)
    E_ref = get_energy(mps_reference, mpo_original, bdim=1024)

    # Get bdim=400 energy in original representation
    mps_original = get_hf_mps(nele=nele,
                              sz=sz,
                              norbs=norbs,
                              bdim=400,
                              e0=e0)
    E_original = get_energy(mps_original, mpo_original, bdim=400)

    # Get localized and reordered HF MPS and MPO
    localize_orbitals(mf, method='meta-lowdin')
    h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    g2e = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff),
                        norbs)
    h1e, g2e, order = apply_fiedler_ordering(h1e, g2e)
    fd = FCIDUMP(pg='c1', n_sites=norbs, n_elec=nele, twos=sz, ipg=0,
                 uhf=False, h1e=h1e, g2e=g2e, const_e=e0)
    mpo_reordered = Hamiltonian(fd, flat=True).build_qc_mpo()
    opts = {'cutoff': 1E-16,
            'max_bond_dim': -1}
    mpo_reordered = MPS(tensors=mpo_reordered.tensors, const=fd.const_e,
                        opts=opts).to_sparse()

    # Get bdim=400 energy in reordered and localized representation
    occ = [0]*norbs
    for i, o in enumerate(order):
        if o < nele // 2:
            occ[i] = 2

    mps_reordered = get_hf_mps(nele=nele,
                               sz=sz,
                               norbs=norbs,
                               bdim=400,
                               e0=e0,
                               occ=occ)
    E_reordered = get_energy(mps_reordered, mpo_reordered, bdim=400)

    print(f"Reference energy: {E_ref}")
    print(f"Reference bdim: {mps_reference.bond_dim}")
    print("Absolute error at bdim=400")
    print(f"Original basis: {np.abs(E_original - E_ref)}")
    print(f"Localized and reordered basis: {np.abs(E_reordered - E_ref)}")
