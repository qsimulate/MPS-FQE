# MPS-FQE Examples

- fqe_to_mps.py
  - Calculates the expectation value of a sparse operator with FQE and MPS-FQE with bra = ket and bra != ket.
- N2_asp.py
  - Performs adiabatic state preparation on N<sub>2</sub> in a minimal basis performed with exact statevector evolution and TD-DMRG under truncated bond dimension.
- N2_imag.py
  - Performs imaginary time propagation on N<sub>2</sub> in a minimal basis performed with exact statevector evolution and TD-DMRG under truncated bond dimension.
- hydrogen_sheet.py
  - Performs DMRG calculations on a 10-member Hydrogen sheet with different orbital localization and ordering schemes in a minimal basis both exactly and under truncated bond dimension.
- adapt.py
  - Performs an ADAPT-VQE simulation on a 4-member Hydrogen chain under truncated bond dimension using RK4 propagation.

```
@article{mps_fqe_2023,
    author       = {Justin Provazza, Klaas Gunst, Huanchen Zhai, Garnet K.-L. Chan,
    		    Toru Shiozaki, Nicholas C. Rubin, and Alec F. White},
    title        = {Fast emulation of fermionic circuits with matrix product states},
    month        = {December},
    year         = {2023},
    url          = {https://doi.org/10.48550/arXiv.2312.17657}
    }
```
