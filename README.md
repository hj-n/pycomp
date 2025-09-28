# PyComp

A Python comparison utility package.

## Prerequisites

`pycomp` uses CuPy and CuML for GPU computations.
For CUDA 12.2:

```sh
pip install --upgrade --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==25.8.*" \
    "dask-cudf-cu12==25.8.*" \
    "cuml-cu12==25.8.*" \
    "cugraph-cu12==25.8.*" \
    "nx-cugraph-cu12==25.8.*" \
    "cuxfilter-cu12==25.8.*" \
    "cucim-cu12==25.8.*" \
    "pylibraft-cu12==25.8.*" \
    "raft-dask-cu12==25.8.*" \
    "cuvs-cu12==25.8.*"\
    "pylibcugraph-cu12==25.8.*"
```

This requires `cuda-toolkit`, installable e.g. via `conda`:
```sh
conda install nvidia/label/cuda-12.2.2::cuda-toolkit
```

## Installation

```bash
pip install pycomp
```

## Usage

```python
import pycomp

# Usage examples will be added as the package develops
```



## Test

To check whether the functions work properly for labeled datasets, run:

```bash
pytest test/test_labeled_datasets.py -v -s
```

