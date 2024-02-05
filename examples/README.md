# MPS-FQE Examples

- N2_asp.py
  - Performs adiabatic state preparation on N$$_2$$ in a minimal basis performed with exact statevector evolution and TD-DMRG under truncated bond dimension.
- N2_imag.py
  - Performs imaginary time propagation on N$$_2$$ in a minimal basis performed with exact statevector evolution and TD-DMRG under truncated bond dimension.
- hydrogen_sheet.py
  - Performs DMRG calculations on a 10-member Hydrogen sheet with different orbital localization and ordering schemes in a minimal basis both exactly and under truncated bond dimension.
- adapt.py
  - Performs an ADAPT-VQE simulation on a 4-member Hydrogen chain under truncated bond dimension using RK4 propagation.


## Collaborators
__QSimulate__:\
Alec F. White, Justin Provazza, Klaas Gunst

__Google__:\
Nicholas C. Rubin

__California Institute of Technology__:\
Huanchen Zhai

## How to cite
When using MPS-FQE for research projects, please cite:

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
