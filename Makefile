PROJECT_NAME = altfisica

# NVCC is path to nvcc. Here it is assumed /opt/cuda is on one's PATH.
# CC is the compiler for C++ host code.

NVCC = nvcc
CC = clang

CUDAPATH = /opt/cuda

BUILD_DIR = build
# note that nvcc defaults to 32-bit architecture. thus, force C/LFLAGS to comply.
# you could also force nvcc to compile 64-bit with -m64 flag. (and remove -m32 instances)

CFLAGS = -c -m64 -I$(CUDAPATH)/include -I/opt/cuda/samples/common/inc/
NVCCFLAGS = -c -I$(CUDAPATH)/include -I/opt/cuda/samples/common/inc/
LFLAGS = -m64 -L$(CUDAPATH)/lib -lcuda -lcudart -lm

all: build clean

build: build_dir gpu
	$(NVCC) $(LFLAGS) -o $(BUILD_DIR)/$(PROJECT_NAME) *.o

build_dir:
	mkdir -p $(BUILD_DIR)

gpu:
	$(NVCC) $(NVCCFLAGS) *.cu

clean:
	rm -f *.o

run:
	./$(BUILD_DIR)/$(PROJECT_NAME)
