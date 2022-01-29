export CUDA_HOME=/misc/software/cuda/cuda-10.0.130-cudnn-7.5.0/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/misc/software/cuda/cuda-10.0.130-cudnn-7.5.0/
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
python3 setup.py build_ext --inplace
