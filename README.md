# MPS-FQE
The Matrix Product State Fermionic Quantum Emulator (MPS-FQE) delivers a matrix product state backend to the [Openfermion-FQE](https://github.com/quantumlib/OpenFermion-FQE) fermionic circuit simulator.

[![Tests](https://github.com/qsimulate/MPS-FQE/workflows/Tests/badge.svg)](https://github.com/qsimulate/MPS-FQE/actions/workflows/test.yml)

## Getting Started
Installing MPS-FQE can be done by executing the `install.sh` script in the current directory.

MPS-FQE depends on [block2](https://github.com/block-hczhai/block2-preview/) v>=0.5.2, [pyblock3](https://github.com/block-hczhai/pyblock3-preview/) v>=0.2.9rc4, and [Openfermion-FQE](https://github.com/quantumlib/OpenFermion-FQE) v >= 0.3.0.


All submissions, including submissions by project members, require review. 
We use GitHub pull requests for this purpose. Consult GitHub Help for more information on using pull requests. 
Furthermore, please make sure your new code comes with extensive tests! We use automatic testing to 
make sure all pull requests pass tests and do not decrease overall test coverage by too much. 
Make sure you adhere to our style guide. Just have a look at our code for clues. 
We mostly follow PEP 8 and use the corresponding linter to check for it. 
Code should always come with documentation, which is generated automatically and can be found here.

We use Github issues for tracking requests and bugs. 

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
