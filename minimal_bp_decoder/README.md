# Minimal BP Decoder Python Package (myldpc)

This package provides a minimal Python interface to a C++ LDPC Belief Propagation (BP) decoder using Cython based on Joshcka's implementation.

## What's New?
- Added oscillation statistics for BP decoding. After decoding, you can now access the number of oscillations encountered during the process.

## Build Instructions

1. Ensure you have Python, Cython, numpy, and scipy installed.
2. Build the extension:

```bash
cd minimal_bp_decoder
python setup.py build_ext --inplace
```

## Using the Compiled Package

After building, if you want to use the compiled package, make sure to add the build directory (where the compiled files are located) to your `PYTHONPATH`:

```bash
export PYTHONPATH=<project_dir>/bpGPU_SF/minimal_bp_decoder/src_python:$PYTHONPATH
```

Replace `<project_dir>` with the actual path if your project is located elsewhere.

## Usage Example

```python
from myldpc.bp_decoder import BpDecoder
import numpy as np

# The rest are same as in Joshcka's BP decoder. 
# Initialization
bpd = BpDecoder(
                    pcm, #the parity check matrix
                    error_channel=priors, # the error rate on each bit
                    max_iter=i_max, #the maximum iteration depth for BP
                    bp_method="ms", #BP method.
                    ms_scaling_factor=0)
bpd.decode(something)
# after decoding, to access oscillations:
self.bpd.oscillation_counts
