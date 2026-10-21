# Installation

The environment definitions target Python 3.10 on 64-bit Linux or Windows
(x86_64). Choose one definition; both create an environment named `celltuneppi`.

## NVIDIA GPU with CUDA 12.4

Run from the repository directory:

```shell
conda env create --file environment.yml
conda activate celltuneppi
```

This installs PyTorch 2.6.0 with its CUDA 12.4 build from the official PyTorch
wheel index. A compatible NVIDIA GPU and driver are required for CUDA execution.
PyTorch resolves its platform-specific runtime dependencies; the environment
file does not list NVIDIA runtime libraries individually.

## CPU

As an alternative, run:

```shell
conda env create --file environment_cpu.yml
conda activate celltuneppi
```

Model entry points default to CUDA. Add `--device cpu` when using the CPU
environment.

## Check the installation

```shell
python -m pip check
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
python scripts/CellTunePPI_train_cv_test.py --help
```

For example, run independent evaluation with the packaged checkpoint:

```shell
python scripts/CellTunePPI_train_cv_test.py --mode test --expr-file example_data/HCT116.h5ad --esm2-file example_data/esm2_embedding.pt --test-labels-tsv example_data/HCT116_ppi_train.tsv --ckpt-path model/model_trained_by_HEK293.pt --curve-output-dir result/hct116_test
```

Add `--device cpu` to this command for CPU execution.

## Dependency scope

The files pin the eight core packages used for model execution and H5AD
preprocessing: PyTorch, NumPy, Pandas, SciPy, AnnData, h5py, Matplotlib, and
Scanpy. Numba, llvmlite, scikit-learn, and UMAP are also pinned to the versions
used in the validated preprocessing environment. Other required dependencies
are resolved automatically by pip.

ESM2 vectors are loaded from the supplied PT file, so running these scripts
does not require installing a protein language model extraction package.

These are application environment definitions. Exact reproduction of a
historical training run also depends on the input files, training options,
hardware, resolved dependencies, and numerical execution settings.

The CPU checks reported for this release used the existing Python 3.10.4
environment with the same core library versions. Fresh installation from
these two environment definitions and CUDA execution have not been tested
locally.

PyTorch build options follow the
[official PyTorch 2.6.0 installation instructions](https://pytorch.org/get-started/previous-versions/#v260).
