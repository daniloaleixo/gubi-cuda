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
CUDAPATH = $(shell which nvcc | sed 's!/bin/nvcc!!')

NVCC = nvcc
CC = gcc

PROJECT_NAME = altfisica

NVCCFLAGS = --compiler-options='-Wno-unused-result -march=native -O3' -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc/
LFLAGS = -m64 -L$(CUDAPATH)/lib -lcuda -lcudart -lm

all: build

build:
	$(NVCC) -o $(PROJECT_NAME) $(NVCCFLAGS) $(LFLAGS) *.cu

clean:
	rm -f $(PROJECT_NAME)

.PHONY: all build clean

