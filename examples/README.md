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
@article{mps_fqe_2024,
  title={Fast emulation of fermionic circuits with matrix product states},
  author={Provazza, Justin and Gunst, Klaas and Zhai, Huanchen and Chan, Garnet K-L and Shiozaki, Toru and Rubin, Nicholas C and White, Alec F},
  journal={Journal of Chemical Theory and Computation},
  volume={20},
  number={9},
  pages={3719--3728},
  year={2024},
  publisher={ACS Publications}
}
```
