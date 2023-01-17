import numpy as np
from mps_fqe.wavefunction import MPSWavefunction
import fqe

wfn = fqe.Wavefunction([[2, 0, 2], [3, 1, 2]])
# wfn = fqe.Wavefunction([[3, 1, 3]])
wfn.set_wfn(strategy="random")
wfn.print_wfn()

mpswavefunction = MPSWavefunction.from_fqe_wavefunction(wfn)
# mpswavefunction.print_wfn()
mpswavefunction.to_fqe_wavefunction().print_wfn()

mpswavefunction = mpswavefunction.canonicalize(0)
# mpswavefunction.print_wfn()
mpswavefunction.to_fqe_wavefunction().print_wfn()

mpswavefunction, merror = mpswavefunction.compress(max_bond_dim=2)
# mpswavefunction.print_wfn()
print(merror)
mpswavefunction.to_fqe_wavefunction().print_wfn()
