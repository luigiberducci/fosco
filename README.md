<h2 align="center">
FOSCo: FOrmal Synthesis of COntrol Barrier Functions
</h2>

<p align="center">
<a href="https://opensource.org/license/bsd-3-clause/"><img alt="License" src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg"></a>
<a href="https://python.org"><img alt="Python 3.10" src="https://img.shields.io/badge/python-3.10-blue.svg"></a>
<a href="https://github.com/luigiberducci/cegis_cbf/actions/workflows/tests-on-push.yml/badge.svg"><img alt="Workflow Status" src="https://github.com/luigiberducci/cegis_cbf/actions/workflows/tests-on-push.yml/badge.svg"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

Learner-verifier framework for synthesis of Control Barrier Functions (CBFs) 
for
general (nonlinear) control-affine systems.

We use a counterexample-guided inductive synthesis (CEGIS) approach to
learn a CBF that guarantees forward invariance with respect to a given
system and unsafe set.

![Example CBF Single-Integrator](docs%2Fsingle_integrator.gif)

## :wrench: Installation 
The code is written in Python 3.10 and uses [PyTorch](https://pytorch.org/) for
learning a CBF.
We recommend using a virtual environment.

To install the required dependencies, run
```bash
pip install -r requirements.txt
```

## :rocket: Examples 
We provide a simple example for a single-integrator system in 
[`run_example.py`](run_example.py).

To run the example, run
```bash
python run_example.py
```

## :warning: Disclaimer
This is a research prototype, tailored for CBF and built on top of [FOSSIL](https://github.com/oxford-oxcav/fossil).
Our implementation aims to refactor the original codebase and keep the minimal functionality required for CBF synthesis.

We invite to refer to the original codebase for synthesis of general Lyapunov certificates.
