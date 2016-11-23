#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#define SIZE 3

typedef struct {
  float ***data;
  int width, height;
} Imagem;

Imagem* new_image(int w, int h) {
  Imagem *res = (Imagem*)malloc(3*w*h*sizeof(float));
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      for (int k = 0; k < 3; k++) {
        res->data[i][j][k] = 0;
      }
    }
  }
  res->width = w;
  res->height = h;
  return res;
}

#define PI acosf(-1)

__global__ void kernel(float *R, float *G, float *B, float *nR, float *nG, float *nB) {
	int x = blockIdx.x;
	int y = blockIdx.y;

  float theta = G[y*SIZE + x]*2*PI;

  float Rx = R[y*SIZE + x]*sinf(theta);
  float Ry = R[y*SIZE + x]*cosf(theta);

  float Bx = -B[y*SIZE + x]*sinf(theta);
  float By = -B[y*SIZE + x]*cosf(theta);

  int xx = 0;
  int yy = 0;

  if (Rx > 0) {
    if (x < SIZE - 1) {
      xx = 1;
    }
  } else {
    if (x > 0) {
      xx = -1;
    }
  }
  if (Ry > 0) {
    if (y < SIZE - 1) {
      yy = 1;
    }
  } else {
    if (y > 0) {
      yy = -1;
    }
  }

    printf("[%d %d] = (%f, %f, %f)\nAng G: %f Rx: %f Ry: %f Bx: %f By: %f\n", x, y, R[y*SIZE + x], G[y*SIZE + x], B[y*SIZE + x], theta, Rx, Ry, Bx, By);
    printf("[%d, %d] -> [%d, %d]\n", x, y, x + xx, y);
    printf("[%d, %d] -> [%d, %d]\n", x, y, x, y + yy);


  float deltaRx = (1 - R[y*SIZE + (x + xx)])*Rx/4.0;
  float deltaRy = (1 - R[(y + yy)*SIZE + x])*Ry/4.0;

  float deltaBx = (1 - B[y*SIZE + (x - xx)])*Bx/4.0;
  float deltaBy = (1 - B[(y - yy)*SIZE + x])*By/4.0;

  if (xx != 0) {
    atomicAdd(&nR[y*SIZE + (x + xx)], deltaRx);
    atomicAdd(&nB[y*SIZE + (x - xx)], deltaBx);
  }
  if (yy != 0) {
    atomicAdd(&nR[(y + yy)*SIZE + x], deltaRy);
    atomicAdd(&nB[(y - yy)*SIZE + x], deltaBy);
  }
	
  return;
}

__global__ void kernel2(float *nR, float *nG, float *nB, float *R, float *G, float *B) {
	int x = blockIdx.x;
	int y = blockIdx.y;
	
	if (nR[x] > 1) {
		float tmp = nR[x] - 1;
		if (x == 1) {
			atomicAdd(&nR[x + 1], tmp/2);
			atomicAdd(&nR[x + SIZE], tmp/2);
		} else if (x == SIZE - 1) {
			atomicAdd(&nR[x - 1], tmp/2);
			atomicAdd(&nR[x + SIZE], tmp/2);
		} else if (x == SIZE*SIZE - 32 - 1) {
			atomicAdd(&nR[x + 1], tmp/2);
			atomicAdd(&nR[x - SIZE], tmp/2);
		} else if (x == SIZE*SIZE - 1) {
			atomicAdd(&nR[x - 1], tmp/2);
			atomicAdd(&nR[x - SIZE], tmp/2);
		}
	}
}

// the wrapper around the kernel call for main program to call.
extern "C" void kernel_wrapper(int num_procs, float *R, float *G, float *B, float *nR, float *nG, float *nB) {
  dim3 image_size(SIZE, SIZE);
	kernel<<<image_size, 1>>>(R, G, B, nR, nG, nB);
}

// the wrapper around the kernel call for main program to call.
extern "C" void kernel2_wrapper(int num_procs, float *nR, float *nG, float *nB, float *R, float *G, float *B) {
  dim3 image_size(SIZE);
	kernel<<<image_size, 1>>>(nR, nG, nB, R, G, B);
}

int main(int argc, char const *argv[]) {	
  if (argc != 5) {
    printf("Uso:\n");
    printf("%s entrada saida num_iters num_procs", argv[0]);
    return 0;
  }

  int iters = atoi(argv[3]);
  int num_procs = atoi(argv[4]);

  srand(time(NULL));

  float *R = (float*)malloc(SIZE*SIZE*sizeof(float));
  float *G = (float*)malloc(SIZE*SIZE*sizeof(float));
  float *B = (float*)malloc(SIZE*SIZE*sizeof(float));
  float *gR, *gG, *gB, *nR, *nG, *nB;

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      R[i*SIZE + j] = ((float)rand()/RAND_MAX);
      G[i*SIZE + j] = ((float)rand()/RAND_MAX);
      B[i*SIZE + j] = ((float)rand()/RAND_MAX);
    }
  }

  cudaMalloc((void**)&gR, SIZE*SIZE*sizeof(float));
  cudaMalloc((void**)&gG, SIZE*SIZE*sizeof(float));
  cudaMalloc((void**)&gB, SIZE*SIZE*sizeof(float));
  cudaMalloc((void**)&nR, SIZE*SIZE*sizeof(float));
  cudaMalloc((void**)&nG, SIZE*SIZE*sizeof(float));
  cudaMalloc((void**)&nB, SIZE*SIZE*sizeof(float));

  cudaMemcpy(gR, R, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gG, G, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gB, B, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(nR, R, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(nG, G, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(nB, B, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
    
	assert(cudaGetLastError() == cudaSuccess);
	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	clock_t start, stop;
	
	for(int i = 0; i < iters; i++) {
		printf("========= Iteracao %d\n", i+1);
		
		start = clock();
    // bagulhos aqui
    kernel_wrapper(num_procs, gR, gG, gB, nR, nG, nB);
		stop = clock();

    cudaMemcpy(R, gR, SIZE*SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(G, gG, SIZE*SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, gB, SIZE*SIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(nR, R, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nG, G, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nB, B, SIZE*SIZE*sizeof(float), cudaMemcpyHostToDevice);
    assert(cudaGetLastError() == cudaSuccess);
	}

  cudaDeviceSynchronize();
	
	return 0;
}
