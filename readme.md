# BP-SF Decoder for qLDPC codes

This repository contains the implementation and examples for the BP-SF decoder used in:

> **Fully Parallelized BP Decoding for Quantum LDPC Codes Can Outperform BP-OSD** (HPCA 2026).

The codebase provides a parallelized belief-propagation (BP) decoder implementation, helper scripts for circuit-level experiments, and a minimal Cython extension (based on the [`ldpc`](https://github.com/quantumgizmos/ldpc) project) used for performance-critical routines.

## 1. Requirements

- Python 3.11 (other 3.x versions may work but are untested)
- Recommended: NVIDIA GPU with CUDA to test GPU-accelerated decoding (optional)
- Build tools for compiling the Cython extension: a C/C++ compiler compatible with your Python distribution

```bash
python -m venv .venv
source .venv/bin/activate 
pip install cython numpy scipy stim setuptools matplotlib
pip install cudaq-qec   # optional: GPU decoding support
pip install ldpc        # optional: baseline BP/BP-OSD 
```

### Build `myldpc` Cython extension

The repository includes a small Cython-based extension in `minimal_bp_decoder` adapted from the [`ldpc`](https://github.com/quantumgizmos/ldpc) project. Build it in-place to allow importing the extension directly from the repo.

Run the build from the project root:

```bash
export PYTHONPATH=./src_python:$PYTHONPATH
cd minimal_bp_decoder
python setup.py build_ext --inplace
```

### Usage Example

```python
from myldpc.bp_decoder import BpDecoder
import numpy as np

# The usage is the same as in Joschka's BP decoder. 
# https://github.com/quantumgizmos/ldpc

# Initialization
bpd = BpDecoder(
                    pcm, #the parity check matrix
                    error_channel=priors, # the error rate on each bit
                    max_iter=i_max, #the maximum iteration depth for BP
                    bp_method="ms", #BP method.
                    ms_scaling_factor=0)
bpd.decode(something)
# after decoding, to access oscillations:
osc_counts = bpd.oscillation_counts 
```

## 2. Run Tests

```bash
sh bpsf_circ.sh  # test BP-SF on circuit-level simulations
sh bposd_circ.sh # test BP-OSD on circuit-level simulations (requires ldpc installed)
sh time.sh       # compare decoding speeds (requires CUDAQ-QEC installed)
```

After running these scripts, the LER and decoding time data will be saved in the `data` directory. You can generate the plots by running the cells in `plots.ipynb`.

Since running all tests will be very time-consuming (days on a 16 cores machine), the test scripts only include main results. You can append more tests with different parameters in the shell scripts to reproduce all results in the paper.
