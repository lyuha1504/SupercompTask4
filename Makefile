GPU_FLAGS=-O3 -std=c++11
MPI_CUDA_FLAGS=-I/opt/ibm/spectrum_mpi/include -L/opt/ibm/spectrum_mpi/lib -lmpiprofilesupport -lmpi_ibm

M=511
N=512
P=2

all:	Task4 Task4_cuda_only

Task4:	Task4.cu
	nvcc $(GPU_FLAGS) $(MPI_CUDA_FLAGS) Task4.cu -o Task4

Task4_cuda_only:	Task4_cuda_only.cu
	nvcc $(GPU_FLAGS) Task4_cuda_only.cu -o Task4_cuda_only

submit-mpi-cuda:	Task4
	bsub -n $(P) -W 00:30 -gpu "num=2" -R "span[ptile=2]" -oo stdout_$(P)_$(M)_$(N).txt -eo error.txt OMP_NUM_THREADS=1 mpiexec ./Task4 $(M) $(N)

submit-cuda-only:	Task4_cuda_only
	bsub -W 00:30 -gpu "num=1" -R "span[ptile=1]" -oo stdout_0_$(M)_$(N).txt -eo error.txt OMP_NUM_THREADS=1 ./Task4_cuda_only $(M) $(N)
