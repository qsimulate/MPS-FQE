import fqe
from openfermion import FermionOperator
from openfermion.utils import hermitian_conjugated
from mps_fqe.wavefunction import MPSWavefunction


if __name__ == '__main__':
    # System parameters
    nele = 4
    sz = 0
    norbs = 4
    # MPS parameters
    mbd = 20
    cutoff = 1e-14

    # Generate random FQE wavefunction instance
    fqe_wfn = fqe.Wavefunction([[nele, sz, norbs]])
    fqe_wfn.set_wfn(strategy="random")
    # Generate MPS-FQE wavefunction instance from FQE wavefunction
    mps_wfn = MPSWavefunction.from_fqe_wavefunction(fqe_wfn,
                                                    max_bond_dim=mbd,
                                                    cutoff=cutoff)

    # Generate Openfermion operator
    of_operator = FermionOperator("1^ 3")
    of_operator += hermitian_conjugated(of_operator)
    # Use it to generate FQE operator
    fqe_operator = fqe.sparse_hamiltonian.SparseHamiltonian(of_operator)

    # Expectation value
    print("<psi|o|psi>:\nFQE:{}\nMPS-FQE:{}".format(
        fqe_wfn.expectationValue(fqe_operator),
        mps_wfn.expectationValue(fqe_operator))
          )

    fqe_wfn_evolved = fqe_wfn.time_evolve(1, fqe_operator)
    mps_wfn_evolved = mps_wfn.time_evolve(1, fqe_operator)

    # Bra state not equal to ket state
    print("<psi|o exp(-i o)|psi>:\nFQE:{}\nMPS-FQE:{}".format(
        fqe_wfn_evolved.expectationValue(fqe_operator,
                                         brawfn=fqe_wfn),
        mps_wfn_evolved.expectationValue(fqe_operator,
                                         brawfn=mps_wfn))
          )
