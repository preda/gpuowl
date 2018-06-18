all: cudaOwl fftbench

cudaOwl: cudaOwl.cu
	nvcc -O2 -o cudaOwl cudaOwl.cu -lcufft

fftbench: fftbench.cu
	nvcc -O2 -o fftbench fftbench.cu -lcufft
