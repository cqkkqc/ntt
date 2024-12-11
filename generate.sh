mkdir build
nvcc ntt.cu -o build/ntt
nvcc gpuinfo.cu -o build/gpuinfo