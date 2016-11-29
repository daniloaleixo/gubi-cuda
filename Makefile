#
#  MAC0431 - Introducao a Programacao Paralela e Distribuida
#
# Fisica Alternativa
#
# Bruno Endo       - 7990982
# Danilo Aleixo    - 7972370
# Gustavo Caparica - 7991020
#

# NOTA: modifique para o caminho de instalacao do CUDA na sua maquina
CUDAPATH = $(which nvcc | sed 's!/bin/nvcc!!')

NVCC = nvcc
CC = gcc

PROJECT_NAME = altfisica

CFLAGS = -O3 -c -m64 -I$(CUDAPATH)/include -I/opt/cuda/samples/common/inc/
NVCCFLAGS = -O3 -c -I$(CUDAPATH)/include -I/opt/cuda/samples/common/inc/
LFLAGS = -m64 -L$(CUDAPATH)/lib -lcuda -lcudart -lm

all: build clean

build: gpu
	$(NVCC) $(LFLAGS) -o $(PROJECT_NAME) *.o

gpu:
	$(NVCC) $(NVCCFLAGS) *.cu

clean:
	rm -f *.o
