echo "Setup Isaac Sim 5.1 Conda environment (pip-based install)."

export PYTHONPATH_PREV=$PYTHONPATH
export LD_LIBRARY_PATH_PREV=$LD_LIBRARY_PATH

# Isaac Sim 5.1 is installed via pip, so no ISAACSIM_PATH or setup_conda_env.sh needed.
# The isaacsim package is importable directly from the conda env's site-packages.

# Set TORCH_CUDA_ARCH_LIST for Blackwell (SM_120) support
export TORCH_CUDA_ARCH_LIST="12.0"
